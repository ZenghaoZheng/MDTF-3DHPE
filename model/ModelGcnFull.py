import torch
import torch.nn as nn
from model.modules.mlp import MLP
from timm.models.layers import DropPath
from einops import rearrange, repeat
from functools import partial
from model.modules.mamba import Mamba



class SpatialEmbedding(nn.Module):
    def __init__(self, dim, channels=6, norm_layer=nn.LayerNorm, joints=17 ):
        super().__init__()
        self.forward_mlp = nn.Linear(channels, dim, bias=True)
        self.backward_mlp = nn.Linear(channels, dim, bias=True)
        self.fusionmlp = nn.Linear(dim, dim, bias=True)
        self.norm = norm_layer(dim)

    def forward(self, x):
        assert len(x.shape) == 3
        vector, angles = self.calculate_angles(x)
        concat_input = torch.cat((x, vector, angles), dim=-1)
        fusion = self.forward_mlp(concat_input)

        x_back = x.flip([1])
        vector_back, angles_back = self.calculate_angles(x_back)
        concat_input_back = torch.cat((x_back, vector_back, angles_back), dim=-1)
        fusion_back = self.forward_mlp(concat_input_back)

        fusion = (fusion + fusion_back.flip([1])) / 2
        output = self.fusionmlp(self.norm(fusion))
        return output


    def calculate_angles(self, joint_positions):
        bf, j, _ = joint_positions.shape

        angles = torch.zeros(bf, j, 1, device=joint_positions.device)
        vector = torch.zeros(bf, j, 2, device=joint_positions.device)
        for i in range(1, j):
            x_diff = joint_positions[:, i, 0] - joint_positions[:, i - 1, 0]
            y_diff = joint_positions[:, i, 1] - joint_positions[:, i - 1, 1]
            vector[:, i, 0] = x_diff
            vector[:, i, 1] = y_diff
            angles[:, i, 0] = torch.atan2(y_diff, x_diff) / torch.pi
        return vector, angles



class MotionMambaBlock(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, d_state, expand, bimamba_type, if_divide_out, init_layer_scale, drop=0., \
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.mamba = Mamba(dim, d_state=d_state, d_conv=4, expand=expand, bimamba_type=bimamba_type,
                           if_divide_out=if_divide_out, init_layer_scale=init_layer_scale)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mamba(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SpatialGcnMamba(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, d_state, expand, bimamba_type, if_divide_out, init_layer_scale, drop=0., \
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.gcnmamba = Mamba(dim, d_state=d_state, d_conv=4, expand=expand, bimamba_type=bimamba_type,
                           if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, mode='spatial')
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.mambanorm1 = norm_layer(dim)
        self.mambanorm2 = norm_layer(dim)


    def forward(self, x):

        x = x + self.drop_path(self.gcnmamba(self.mambanorm1(x)))
        x = x + self.drop_path(self.mlp(self.mambanorm2(x)))

        return x


class MotionMamba(nn.Module):
    def __init__(self, args):
        super().__init__()

        st_depth = args.st_depth
        self.st_depth = st_depth
        embed_dim = args.channel
        mlp_hidden_dim = args.channel * 2
        d_state = args.d_state
        bimamba_type = args.bimamba_type
        if_divide_out = args.if_divide_out
        init_layer_scale = None
        num_joints = args.n_joints
        expand = args.expand
        drop_path_rate = 0.1
        drop_rate = 0.

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # self.Spatial_patch_to_embedding = nn.Linear(3, embed_dim)
        self.SpatialEmbedding = SpatialEmbedding(dim=embed_dim, channels=6)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, args.n_frames, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, st_depth)]

        self.STEblocks = nn.ModuleList([
            SpatialGcnMamba(
                embed_dim, mlp_hidden_dim, d_state, expand, bimamba_type, if_divide_out, init_layer_scale, drop=drop_rate, \
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(st_depth)])

        self.TTEblocks = nn.ModuleList([
            MotionMambaBlock(
                dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, d_state=d_state, expand=expand, bimamba_type=bimamba_type, \
                if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, drop=drop_rate, drop_path=dpr[i], \
                norm_layer=norm_layer, )
            for i in range(st_depth)])

        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)
        self.line_pose = args.line_pose

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3),
        )

    def forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c  -> (b f) n c')

        x = self.SpatialEmbedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)
        x = rearrange(x, '(b n) f c -> b f n c', n=n)


        for i in range(1, self.st_depth):
            x = rearrange(x, 'b f n c -> (b f) n c')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]

            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n c -> (b n) f c', b=b)
            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)

        x = self.head(x)
        x = x.view(b, f, -1, 3)

        return x

