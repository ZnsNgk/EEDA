from os import sep
from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import numpy as np

__all__ = [
    'p2t_base', 'p2t_eeda_base'
]

class Efficient_Linear_Embedding_Dimensional_Attention(nn.Module):
    # E^2DA (Efficient EDA with Linear Attention)
    def __init__(self, dim, eda_num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=196, **kwargs):

        super().__init__()

        self.dim = dim
        self.num_heads = num_patches
        self.num_patches = num_patches
        self.scale = qk_scale or nn.Parameter(torch.ones(1, self.num_heads), requires_grad=True)

        self.qk = nn.Linear(num_patches, num_patches, False)
        self.qkv_bias = qkv_bias
        self.k_bias = nn.Parameter(torch.zeros(num_patches), requires_grad=True) if qkv_bias else None
        self.q_bias = nn.Parameter(torch.zeros(num_patches), requires_grad=True) if qkv_bias else None
        
        self.v = nn.Linear(num_patches, num_patches, qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_patches, num_patches)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.GELU()
        

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous() # B, C, N
        
        k = self.qk(x)    # B, C, N

        # q = x.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()    # B, n_heads, C, N/n_heads
        # k = k.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()    # B, n_heads, C, N/n_heads
        # v = self.v(x).reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()    # B, n_heads, C, N/n_heads

        v = self.v(x)   # B, C, N

        if self.qkv_bias:
            k_norm = (k + self.k_bias).norm(dim=-1, keepdim=True)   
            q_norm = (k + self.q_bias).norm(dim=-1, keepdim=True)   # ATTENTION: This is k.norm(), because of ||XW_qk^T|| and ||W_qkX^T|| are equal.

            k = k + self.k_bias
            q = x + self.q_bias
        else:
            k_norm = k.norm(dim=-1, keepdim=True)
            q_norm = k.norm(dim=-1, keepdim=True)

            k = k
            q = x
        
        q = q / q_norm  # B, C, N
        k = k / k_norm
        
        kernel_trick = (k * v).sum(dim=-2, keepdim=True) * self.scale
        x = (q * kernel_trick)
        
        # x = x.transpose(1,2).contiguous().reshape(B, C, N)
        x = self.act(x)
        x = self.proj(x).transpose(1,2).contiguous()
        # x = self.act(x)

        return x

class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize//2, stride=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0,2,1).reshape(B, C, H, W).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.reshape(B, C, -1).permute(0,2,1).contiguous()


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
        pool_ratios=[1,2,3,6], HW=None):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t*t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()
        
        self.norm = nn.LayerNorm(dim)
        self.pool_target_size = []
        for pool_ratio in self.pool_ratios:
            self.pool_target_size.append((round(HW/pool_ratio), round(HW/pool_ratio)))

    def forward(self, x, H, W, d_convs=None):
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        for (pool_ratio, l, target_size) in zip(self.pool_ratios, d_convs, self.pool_target_size):
            pool = F.adaptive_avg_pool2d(x_, target_size)     
            # pool = F.adaptive_avg_pool2d(x_, (round(H/pool_ratio), round(W/pool_ratio)))
            pool = pool + l(pool) # fix backward bug in higher torch versions when training
            pools.append(pool.view(B, C, -1).contiguous())
        
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0,2,1).contiguous())
        
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   
        x = x.transpose(1,2).contiguous().reshape(B, N, C).contiguous()
        
        x = self.proj(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratios=[12,16,20,24], num_patches=None, HW=None, use_eda=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop, proj_drop=drop, pool_ratios=pool_ratios, HW=HW)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        if use_eda:
            self.mlp = Efficient_Linear_Embedding_Dimensional_Attention(dim, num_patches=num_patches)
        else:
            self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=drop, ksize=3)
    
    def forward(self, x, H, W, d_convs=None):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, d_convs=d_convs))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class PatchEmbed(nn.Module):
    """ (Overlapped) Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=3, in_chans=3, embed_dim=768, overlap=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        if not overlap:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=kernel_size//2)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape 
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, (H, W)



class PyramidPoolingTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 9, 3], use_eda=False, **kwargs): #
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.embed_dims = embed_dims
        self.use_eda = use_eda

        # pyramid pooling ratios for each stage
        pool_ratios = [[12,16,20,24], [6,8,10,12], [3,4,5,6], [1,2,3,4]]
        
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, kernel_size=7, in_chans=in_chans,
                                       embed_dim=embed_dims[0], overlap=False)

        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
                                    embed_dim=embed_dims[1], overlap=True)
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
                                    embed_dim=embed_dims[2], overlap=True)
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
                                    embed_dim=embed_dims[3], overlap=True)
        
        self.num_patches1 = self.patch_embed1.num_patches
        self.num_patches2 = self.patch_embed2.num_patches
        self.num_patches3 = self.patch_embed3.num_patches
        self.num_patches4 = self.patch_embed4.num_patches
        self.HW1 = int(math.sqrt(self.num_patches1))
        self.HW2 = int(math.sqrt(self.num_patches1))
        self.HW3 = int(math.sqrt(self.num_patches1))
        self.HW4 = int(math.sqrt(self.num_patches1))

        
        self.d_convs1 = nn.ModuleList([nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1, groups=embed_dims[0]) for temp in pool_ratios[0]])
        self.d_convs2 = nn.ModuleList([nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, groups=embed_dims[1]) for temp in pool_ratios[1]])
        self.d_convs3 = nn.ModuleList([nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, groups=embed_dims[2]) for temp in pool_ratios[2]])
        self.d_convs4 = nn.ModuleList([nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1, groups=embed_dims[3]) for temp in pool_ratios[3]])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0


        ksize = 3

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[0], num_patches=self.num_patches1, HW=self.HW1)
            for i in range(depths[0])])
        

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[1], num_patches=self.num_patches2, HW=self.HW2)
            for i in range(depths[1])])

        cur += depths[1]

        
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[2], num_patches=self.num_patches3, HW=self.HW3, use_eda=True if self.use_eda and i>depths[2]//2-1 else False)
            for i in range(depths[2])])

        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[3], num_patches=self.num_patches4, HW=self.HW4, use_eda=True if self.use_eda and i>depths[2]//2-1 else False)
            for i in range(depths[3])])
        
        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.apply(self._init_weights)

        #print(self)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        
        for idx, blk in enumerate(self.block1):
            x = blk(x, H, W, self.d_convs1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, (H, W) = self.patch_embed2(x)

        for idx, blk in enumerate(self.block2):
            x = blk(x, H, W, self.d_convs2)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, (H, W) = self.patch_embed3(x)

        for idx, blk in enumerate(self.block3):
            x = blk(x, H, W, self.d_convs3)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # stage 4
        x, (H, W) = self.patch_embed4(x)

        for idx, blk in enumerate(self.block4):
            x = blk(x, H, W, self.d_convs4)
        
        return x
    
    def forward_features_for_fpn(self, x):
        outs = []

        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        
        for idx, blk in enumerate(self.block1):
            x = blk(x, H, W, self.d_convs1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, (H, W) = self.patch_embed2(x)

        for idx, blk in enumerate(self.block2):
            x = blk(x, H, W, self.d_convs2)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, (H, W) = self.patch_embed3(x)

        for idx, blk in enumerate(self.block3):
            x = blk(x, H, W, self.d_convs3)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # stage 4
        x, (H, W) = self.patch_embed4(x)

        for idx, blk in enumerate(self.block4):
            x = blk(x, H, W, self.d_convs4)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        x = torch.mean(x, dim=1)
        x = self.head(x)

        return x
    
    def forward_for_fpn(self, x):
        return self.forward_features_for_fpn(x)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size)).contiguous()
        out_dict[k] = v

    return out_dict


@register_model
def p2t_base(pretrained=False, **kwargs):
    model = PyramidPoolingTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def p2t_eeda_base(pretrained=False, **kwargs):
    model = PyramidPoolingTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], use_eda=True,
        **kwargs)
    model.default_cfg = _cfg()

    return model

if __name__ == "__main__":
    model = p2t_eeda_base().cuda()
    # print(model)
    img = torch.randn([1, 3, 224, 224]).cuda()
    from thop import profile, clever_format
    with torch.no_grad():
        flops, params = profile(model, inputs=(img,))
        flops, params = clever_format([flops, params], "%.6f")
    from torch_flops import TorchFLOPsByFX
    flops_counter = TorchFLOPsByFX(model)
    # Print the grath (not essential)
    print('*' * 120)
    avg_time = 0.
    flops_counter.graph_model.graph.print_tabular()
    # Feed the input tensor
    with torch.no_grad():
        flops_counter.propagate(img)
    # Print the flops of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
    print('*' * 120)
    result_table = flops_counter.print_result_table()
    # Print the total FLOPs
    total_flops = flops_counter.print_total_flops()
    total_time = flops_counter.print_total_time()
    max_memory = flops_counter.print_max_memory()
    print("num_paras: ", params)