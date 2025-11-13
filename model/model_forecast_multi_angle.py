import torch
import torch.nn as nn
from model.modules.mlp import MLP
from timm.models.layers import DropPath
from einops import rearrange, repeat
from functools import partial
from model.modules.mamba import Mamba


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

        embed_dim = [int(args.channel/(2**i)) for i in range(st_depth)]
        mlp_hidden_dim = [int(args.channel/(2**i))*2 for i in range(st_depth)]


        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.Spatial_patch_to_embedding = nn.Linear(3, args.channel)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, st_depth)]

        self.STEblocks = nn.ModuleList([
            MotionMambaBlock(
                dim=embed_dim[i], mlp_hidden_dim=mlp_hidden_dim[i], d_state=d_state, expand=expand,
                bimamba_type=bimamba_type, if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, drop=drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, )
            for i in range(st_depth)])

        self.TTEblocks = nn.ModuleList([
            MotionMambaBlock(
                dim=embed_dim[i], mlp_hidden_dim=mlp_hidden_dim[i], d_state=d_state, expand=expand, bimamba_type=bimamba_type, \
                if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, drop=drop_rate, drop_path=dpr[i], \
                norm_layer=norm_layer, )
            for i in range(st_depth)])

        self.mlp = nn.ModuleList([
            nn.Linear(embed_dim[i], embed_dim[i+1]) for i in range(st_depth-1)
        ] + [nn.Linear(embed_dim[-1], 6)])
        self.Spatial_norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(st_depth)])
        self.Temporal_norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(st_depth)])
        self.classifier_6 = nn.Linear(embed_dim[-1], 6)
        self.classifier_7 = nn.Linear(embed_dim[-1], 7)

    def forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x = rearrange(x, '(b f) n c -> b f n c', f=f)


        for i in range(self.st_depth):
            x = rearrange(x, 'b f n c -> (b f) n c')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            mlp = self.mlp[i]
            x = steblock(x)
            x = self.Spatial_norm[i](x)
            x = rearrange(x, '(b f) n c -> (b n) f c', b=b)
            x = tteblock(x)
            x = self.Temporal_norm[i](x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)
            if i < self.st_depth - 1:
                x = mlp(x)
        out_6 = self.classifier_6(x)
        out_7 = self.classifier_7(x)
        return out_6, out_7

