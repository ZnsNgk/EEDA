import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class MLP_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        # x = x.permute(0,2,1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x# x.permute(0,2,1)
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or nn.Parameter(torch.Tensor([head_dim ** -0.5]), requires_grad=True)

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   
        x = x.transpose(1,2).contiguous().reshape(B, N, C)

        x = self.proj(x)

        return x

class Cross_Covariance_Attention(nn.Module):
    # XCAtt in XCiT
    # From https://arxiv.org/pdf/2106.09681
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.temp = qk_scale or nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.Hardswish()
        
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # split into query, key and value
        
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1) # Transpose to shape (B, h, C, N)
        v = v.transpose(-2, -1)
        
        q = F.normalize(q, dim=-1, p=2) # L2 Normalization across the token dimension
        k = F.normalize(k, dim=-1, p=2)
        
        attn = (k @ q.transpose(-2, -1)) # Computing the block diagonal cross-covariance matrix
        attn = attn * self.temp # Adjusting the activations scale with temperature parameter
        attn = attn.softmax(dim=-1) # d x d attention map
        x = attn @ v # Apply attention to mix channels per token
        
        x = x.permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
    
class Embedding_Dimensional_Attention(nn.Module):
    # EDA
    # There is a certain probability that you will crash during training and need to RESUME multiple times
    def __init__(self, dim, eda_num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=196, **kwargs):
        super().__init__()
        assert num_patches % eda_num_heads == 0, f"dim {num_patches} should be divided by num_heads {eda_num_heads}."

        self.dim = dim
        self.num_heads = eda_num_heads
        self.num_patches = num_patches
        head_dim = num_patches // eda_num_heads
        self.scale = qk_scale or nn.Parameter(torch.Tensor([head_dim ** -0.5]), requires_grad=True)

        self.q = nn.Linear(num_patches, num_patches, bias=qkv_bias)
        self.kv = nn.Linear(num_patches, num_patches * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(num_patches, num_patches)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.GELU()
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(-2, -1) # B, C, N

        q = self.q(x).reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, N // self.num_heads).permute(2, 0, 3, 1, 4)
    
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale   # B, n_heads, C, C
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v    # B, n_heads, C, N/n_heads

        x = x.transpose(1,2).contiguous().reshape(B, C, N)
        x = self.act(x)
        x = self.proj(x).transpose(1,2)

        return x

class Hydra_Embedding_Dimensional_Attention(nn.Module):
    # HEDA (EDA with Hydra trick)
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=196, **kwargs):

        super().__init__()

        self.dim = dim
        self.num_heads = num_patches
        self.num_patches = num_patches
        head_dim = dim // num_patches
        self.scale = qk_scale or nn.Parameter(torch.Tensor([head_dim ** -0.5]), requires_grad=True)

        self.q = nn.Sequential(nn.Linear(num_patches, num_patches, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(num_patches, num_patches * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_patches, num_patches)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.GELU()
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(-2, -1)
        q = self.q(x)#.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x)#.reshape(B, -1, 2, self.num_heads, N // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        kernel_trick = (k * v).sum(dim=-2, keepdim=True)
        x = (q @ kernel_trick) * self.scale
        
        # x = x.squeeze(-1)
        x = self.act(x)
        x = x.transpose(1,2).contiguous()
        x = self.proj(x).transpose(1,2)
        # x = self.act(x)

        return x

class Linear_Embedding_Dimensional_Attention(nn.Module):
    # LEDA (EDA with Linear Attention)
    def __init__(self, dim, eda_num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=196, **kwargs):

        super().__init__()

        self.dim = dim
        self.num_heads = eda_num_heads
        self.num_patches = num_patches
        self.scale = qk_scale or nn.Parameter(torch.ones(eda_num_heads, 1, 1), requires_grad=True)

        self.q = nn.Sequential(nn.Linear(num_patches, num_patches, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(num_patches, num_patches * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_patches, num_patches)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.GELU()
        

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous() # B, C, N
        
        q = self.q(x).reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, N // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        kernel_trick = (k * v).sum(dim=-2, keepdim=True) * self.scale
        x = (q * kernel_trick)
        
        x = x.transpose(1,2).contiguous().reshape(B, C, N)
        x = self.act(x)
        x = self.proj(x).transpose(1,2)
        # x = self.act(x)

        return x
        
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
        

    def forward(self, x):
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
        x = self.proj(x).transpose(1,2)
        # x = self.act(x)

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
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, (H, W)


class DWConv(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features, out_features, kernel_size=kernel_size, padding=padding, groups=out_features)

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)


        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1).contiguous()
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Efficient_Channel_Attention(nn.Module):
    # ECA in FAN
    # from https://github.com/NVlabs/FAN/blob/master/models/fan.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., linear=False, drop_path=0., 
                 mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm, cha_sr_ratio=1, c_head_num=None, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        num_heads = c_head_num or num_heads
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.cha_sr_ratio = cha_sr_ratio if num_heads > 1 else 1

        # config of mlp for v processing
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = Mlp(in_features=dim//self.cha_sr_ratio, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        self.norm_v = norm_layer(dim//self.cha_sr_ratio)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def _gen_attn(self, q, k):
        q = q.softmax(-2).transpose(-1,-2)
        _, _, N, _  = k.shape
        k = torch.nn.functional.adaptive_avg_pool2d(k.softmax(-2), (N, 1))
        
        attn = torch.nn.functional.sigmoid(q @ k)
        return attn  * self.temperature
    def forward(self, x):
        B, N, C = x.shape

        img_size = int(math.sqrt(N))
        # img_size = 14
        H = img_size
        W = img_size

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        Bv, Hd, Nv, Cv = v.shape
        
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv).contiguous(), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)

        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x#,  (attn * v.transpose(-1, -2)).transpose(-1, -2) #attn
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}
