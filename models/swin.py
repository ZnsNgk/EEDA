import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.fx_features import register_notrace_function
from timm.models.helpers import build_model_with_cfg, named_apply, checkpoint_seq
from timm.models.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert
from timm.models.registry import register_model
from timm.models.vision_transformer import checkpoint_filter_fn, get_init_weights_vit

class Hydra_Emb_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=196):

        super().__init__()
        # assert num_patches % num_heads == 0, f"dim {num_patches} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_patches
        self.num_patches = num_patches
        head_dim = 1
        self.scale = qk_scale or nn.Parameter(torch.Tensor([head_dim ** -0.5]), requires_grad=True)

        self.q = nn.Sequential(nn.Linear(num_patches, num_patches, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(num_patches, num_patches * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_patches, num_patches)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.Hardswish()
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape

        x = x.transpose(-2, -1)
        q = self.q(x).reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, N // self.num_heads).permute(2, 0, 3, 1, 4)
    
        k, v = kv[0], kv[1]
        
        q = q / q.norm()
        k = k / k.norm()
        kernel_trick = (k.transpose(-2, -1) @ v)
        x = (q @ kernel_trick) * self.scale
        
        x = x.transpose(1,2).contiguous().reshape(B, N, C)
        x = self.act(x)
        x = x.transpose(1,2).contiguous().reshape(B, C, N)
        x = self.proj(x).transpose(1,2)
        x = self.act(x)

        return x

class Efficient_Linear_Embedding_Dimensional_Attention(nn.Module):
    # E^2DA (Efficient EDA with Linear Attention)
    def __init__(self, dim, eda_num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=196, **kwargs):

        super().__init__()

        self.dim = dim
        self.num_heads = eda_num_heads
        self.num_patches = num_patches
        self.scale = qk_scale or nn.Parameter(torch.ones(eda_num_heads, 1, 1), requires_grad=True)

        self.qk = nn.Linear(num_patches, num_patches, False)
        self.qkv_bias = qkv_bias
        self.k_bias = nn.Parameter(torch.zeros(num_patches), requires_grad=True) if qkv_bias else None
        self.q_bias = nn.Parameter(torch.zeros(num_patches), requires_grad=True) if qkv_bias else None
        
        self.v = nn.Linear(num_patches, num_patches, qkv_bias)
        self.eps = 1e-8

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_patches, num_patches)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.GELU()
        

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous() # B, C, N
        
        k = self.qk(x)    # B, C, N
        if self.qkv_bias:
            k_norm = (k + self.k_bias).norm()   
            q_norm = (k + self.q_bias).norm()   # ATTENTION: This is k.norm(), because of ||XW_qk^T|| and ||W_qkX^T|| are equal.

            k = k + self.k_bias
            q = x + self.q_bias
        else:
            k_norm = k.norm()
            q_norm = k.norm()

            k = k
            q = x
        
        q = q.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()    # B, n_heads, C, N/n_heads
        k = k.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()    # B, n_heads, C, N/n_heads
        v = self.v(x).reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()    # B, n_heads, C, N/n_heads

        q = q / (q_norm + self.eps)
        k = k / (k_norm + self.eps) 
        
        kernel_trick = (k.transpose(-2, -1) @ v) * self.scale     # B, n_heads, N/n_heads, N/n_heads
        x = (q @ kernel_trick)      # B, n_heads, C, N/n_heads
        
        x = x.transpose(1,2).contiguous().reshape(B, C, N)
        x = self.act(x)
        x = self.proj(x).transpose(1,2)
        # x = self.act(x)

        return x

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_eda=False, num_patches=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        if use_eda:
            self.mlp = Efficient_Linear_Embedding_Dimensional_Attention(dim, num_patches=num_patches)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        _assert(L == H * W, "input feature has wrong size")

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, "input feature has wrong size")
        _assert(H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even.")

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        head_dim (int): Channels per head (dim // num_heads if not set)
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
            self, dim, out_dim, input_resolution, depth, num_heads=4, head_dim=None,
            window_size=7, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
            drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_eda=False, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.grad_checkpointing = False

        # build blocks
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, head_dim=head_dim,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, use_eda=True if use_eda and i>depth//2-1 else False, **kwargs)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(
            self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg',
            embed_dim=128, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), head_dim=None,
            window_size=7, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True, weight_init='', use_eda=False, **kwargs):
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.patch_grid = self.patch_embed.grid_size

        # absolute position embedding
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        embed_out_dim = embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        layers = []

        for i in range(self.num_layers):
            layers += [BasicLayer(
                dim=embed_dim[i],
                out_dim=embed_out_dim[i],
                input_resolution=(self.patch_grid[0] // (2 ** i), self.patch_grid[1] // (2 ** i)),
                depth=depths[i],
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
                use_eda = True if (use_eda and i>1) else False,
                num_patches=self.patch_embed.num_patches // (4 ** i)
            )]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Multi classification heads
        self.heads = nn.Linear(embed_dim[-1], num_classes)

        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        if self.absolute_pos_embed is not None:
            trunc_normal_(self.absolute_pos_embed, std=.02)
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.heads(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def swin_base_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model = SwinTransformer(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), use_eda=False, **kwargs)

    return model

@register_model
def swin_eeda_base_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model = SwinTransformer(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), use_eda=True, **kwargs)

    return model

if __name__ == "__main__":
    model = swin_eeda_base_patch4_window7_224().cuda()
    # print(model)
    img = torch.randn([1, 3, 224, 224]).cuda()
    from thop import profile, clever_format
    with torch.no_grad():
        flops, params = profile(model, inputs=(img,))
        flops, params = clever_format([flops, params], "%.6f")
    from torch_flops import TorchFLOPsByFX
    # Print the grath (not essential)
    avg_time = 0.
    # Feed the input tensor
    try:
        flops_counter = TorchFLOPsByFX(model)
        print('*' * 120)
        flops_counter.graph_model.graph.print_tabular()
        with torch.no_grad():
            flops_counter.propagate(img)
            print('*' * 120)
            result_table = flops_counter.print_result_table()
            # Print the total FLOPs
            total_flops = flops_counter.print_total_flops()
            total_time = flops_counter.print_total_time()
            max_memory = flops_counter.print_max_memory()
    except Exception as e:
        print(e)
    # Print the flops of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
    
    print("num_paras: ", params)