import torch
import torch.nn as nn
from model.modules.mlp import MLP
from timm.models.layers import DropPath
from einops import rearrange, repeat
from functools import partial
from model.modules.mamba import Mamba


node = [0,1,2,3,2,1,0,4,5,6,5,4,0,7,8,11,12,13,12,11,8,14,15,16,15,14,8,9,10]


def calculate_angles(joint_positions):
    bf, j, _ = joint_positions.shape

    angles = torch.zeros(bf, j, 1, device=joint_positions.device)

    # 计算角度
    for i in range(1, j):
        # 计算相对向量
        x_diff = joint_positions[:, i, 0] - joint_positions[:, i - 1, 0]
        y_diff = joint_positions[:, i, 1] - joint_positions[:, i - 1, 1]

        # 计算夹角，并将结果转换为度
        angles[:, i, 0] = torch.atan2(y_diff, x_diff) / torch.pi

    return angles


def walkingnode(x):
    assert x.shape[1] == 17
    return x[:,node]


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


class SplitSpatialBlock(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, d_state, expand, bimamba_type, if_divide_out, init_layer_scale, drop, drop_path,act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        self.groups = ((1, 2, 3), (4, 5, 6), (0, 7, 8, 9, 10), (11, 12, 13), (14, 15, 16))
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(17)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.LeftLegMamba = Mamba(dim, d_state=d_state, d_conv=2, expand=expand, bimamba_type=bimamba_type, \
                           if_divide_out=if_divide_out, init_layer_scale=init_layer_scale)
        self.RightLegMamba = Mamba(dim, d_state=d_state, d_conv=2, expand=expand, bimamba_type=bimamba_type, \
                           if_divide_out=if_divide_out, init_layer_scale=init_layer_scale)
        self.LeftArmMamba = Mamba(dim, d_state=d_state, d_conv=2, expand=expand, bimamba_type=bimamba_type, \
                           if_divide_out=if_divide_out, init_layer_scale=init_layer_scale)
        self.RightArmMamba = Mamba(dim, d_state=d_state, d_conv=2, expand=expand, bimamba_type=bimamba_type, \
                           if_divide_out=if_divide_out, init_layer_scale=init_layer_scale)
        self.SpineMamba = Mamba(dim, d_state=d_state, d_conv=2, expand=expand, bimamba_type=bimamba_type, \
                           if_divide_out=if_divide_out, init_layer_scale=init_layer_scale)
        self.fusion = MLP(in_features=17, hidden_features=2 * 17, act_layer=act_layer, drop=drop)

        self.parttemporal = MotionMambaBlock(dim, mlp_hidden_dim, d_state, expand, bimamba_type=bimamba_type,
                                             if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, drop=drop, drop_path=drop_path)

    def forward(self, x):
        b, f, j, c = x.shape
        x = rearrange(x, 'b f j c -> (b f) j c')
        x_ = []
        LeftLeg = x[:, 1:4, :]
        RightLeg = x[:, 4:7, :]
        Spine = x[:, [0, 7, 8, 9, 10], :]
        RightArm = x[:, 11:14, :]
        LeftArm = x[:, 14:17, :]
        LeftLeg = LeftLeg+self.LeftLegMamba(self.norm1(LeftLeg))
        RightLeg = RightLeg+self.RightLegMamba(self.norm2(RightLeg))
        LeftArm = LeftArm+self.LeftArmMamba(self.norm3(LeftArm))
        RightArm = RightArm+self.RightArmMamba(self.norm4(RightArm))
        Spine = Spine+self.SpineMamba(self.norm5(Spine))
        x_.append(LeftLeg)
        x_.append(RightLeg)
        x_.append(Spine)
        x_.append(RightArm)
        x_.append(LeftArm)
        x = torch.cat(x_, dim=1)
        x = x[:, self.inverse_group, :]

        x = x.transpose(1, 2)
        x = x+self.drop_path(self.fusion(self.norm6(x)))
        x = x.transpose(1, 2)

        x = rearrange(x, '(b f) j c -> (b j) f c', f=f)
        x = self.parttemporal(x)
        x = rearrange(x, '(b j) f c -> b f j c', j=j)

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
        self.Spatial_patch_to_embedding = nn.Linear(3, embed_dim)
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
        if self.line_pose:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2, padding=1)
            #self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=2, padding=1)
            #self.conv2d = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3),
        )

    def forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c  -> (b f) n c')


        x = self.Spatial_patch_to_embedding(x)
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

        if self.line_pose:
            # conv1d
            # x = rearrange(x, 'b f n c -> (b f) c n')
            # x = self.conv1d(x)
            # x = rearrange(x, '(b f) c n -> b f n c', b=b)

            # pooling
            x = rearrange(x, 'b f n c -> (b f) c n')
            x = self.pooling(x)
            x = rearrange(x, '(b f) c n -> b f n c', b=b)

            # 对应关节点
            #x = x[:,:,[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],:]
        x = self.head(x)
        x = x.view(b, f, -1, 3)

        return x

