import torch
import torch.nn as nn
from model.modules.mlp import MLP
from timm.models.layers import DropPath
from einops import rearrange, repeat
from functools import partial
from model.modules.mamba import Mamba
from model.modules.graph import GCN

# for h36m
CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9, 14, 11], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7, 4], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}
CONNECTIONS_flip = {6: [7], 7: [6, 8], 8: [7, 9, 2, 5], 2: [1, 8], 1: [0, 2], 5: [8, 4], 4: [5, 3], 9: [8, 16],
                    16:[9, 12, 15], 15:[16,14], 14:[15,13], 12:[16,11], 11:[12,10], 0:[1], 3:[4], 13:[14], 10:[11]}

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
        self.norm_gcn = nn.LayerNorm(dim)
        self.gcn = GCN(dim, dim, num_nodes=17, mode='spatial', connections=CONNECTIONS)


        self.gcnmamba = Mamba(dim, d_state=d_state, d_conv=4, expand=expand, bimamba_type=bimamba_type,
                           if_divide_out=if_divide_out, init_layer_scale=init_layer_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.mambanorm1 = norm_layer(dim)
        self.mambanorm2 = norm_layer(dim)
        self.ts_attn = nn.Linear(dim * 2, 2)

    def forward(self, x):
        x1 = x + self.drop_path(self.gcn(self.norm_gcn(x)))
        x2 = x + self.drop_path(self.gcnmamba(self.mambanorm1(x)))
        alpha = torch.cat([x1, x2], dim=-1)
        alpha = self.ts_attn(alpha)
        alpha = alpha.softmax(dim=-1)
        x = x1 * alpha[:, :, 0:1] + x2 * alpha[:, :, 1:2]
        x = x + self.drop_path(self.mlp(self.mambanorm2(x)))
        return x


class MotionMamba(nn.Module):
    def __init__(self, args):
        super().__init__()

        st_depth = args.st_depth
        self.st_depth = st_depth
        d_state = args.d_state
        bimamba_type = args.bimamba_type
        if_divide_out = args.if_divide_out
        init_layer_scale = None
        num_joints = args.n_joints
        expand = args.expand
        drop_path_rate = 0.1
        drop_rate = 0.
        embed_dim = args.channel
        mlp_hidden_dim = args.channel * 2
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.Spatial_patch_to_embedding1 = nn.Linear(6, args.channel)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, args.n_frames, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, st_depth)]

        self.STEblocks = nn.ModuleList([
            SpatialGcnMamba(
                dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, d_state=d_state, expand=expand, bimamba_type=bimamba_type,
                if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, drop=drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, )
            for i in range(st_depth)])

        self.TTEblocks = nn.ModuleList([
            MotionMambaBlock(
                dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, d_state=d_state, expand=expand, bimamba_type=bimamba_type,
                if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, drop=drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, )
            for i in range(st_depth)])

        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3),
        )


    def forward(self, x):
        b, f, n, _ = x.shape
        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.Spatial_patch_to_embedding1(x)

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

