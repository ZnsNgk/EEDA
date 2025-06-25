import os
import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'poolformer_s': _cfg(crop_pct=0.9),
    'poolformer_m': _cfg(crop_pct=0.95),
}


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
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
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

        # reshape to [B, C, H, W]
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return x

class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        stride = (stride, stride)
        padding = (padding, padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5, use_eda=False, num_patches=None):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_eda:
            self.mlp = Efficient_Linear_Embedding_Dimensional_Attention(dim, num_patches=num_patches)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers, 
                 pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5, use_eda=False, num_patches=None):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        eda = False
        if use_eda and block_idx > layers[index] // 2 - 1:
            eda = True
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value, use_eda=eda, num_patches=num_patches
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class PoolFormer(nn.Module):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and 
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained: 
        for mmdetection and mmsegmentation to load pretrained weights
    """
    def __init__(self, layers, embed_dims=None, 
                 mlp_ratios=None, downsamples=None, 
                 pool_size=3, 
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 fork_feat=False,
                 init_cfg=None, 
                 pretrained=None, 
                 use_eda=False,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=3, embed_dim=embed_dims[0])
        num_patches = [56**2, 28**2, 14**2, 7**2]

        # set the main block in network
        network = []
        for i in range(len(layers)):
            eda = False
            if i > 1 and use_eda:
                eda = True
            stage = basic_blocks(embed_dims[i], i, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value, use_eda=eda, num_patches=num_patches[i])
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model 
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading 
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            
            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out

@register_model
def poolformer_m48(pretrained=False, **kwargs):
    """
    PoolFormer-M48 model, Params: 73M
    """
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    return model

@register_model
def poolformer_eeda_m48(pretrained=False, **kwargs):
    """
    PoolFormer-M48 model, Params: 73M
    """
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, use_eda=True,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    return model

if has_mmseg and has_mmdet:
    """
    The following models are for dense prediction based on 
    mmdetection and mmsegmentation
    """
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s12_feat(PoolFormer):
        """
        PoolFormer-S12 model, Params: 12M
        """
        def __init__(self, **kwargs):
            layers = [2, 2, 6, 2]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, 
                mlp_ratios=mlp_ratios, downsamples=downsamples, 
                fork_feat=True,
                **kwargs)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s24_feat(PoolFormer):
        """
        PoolFormer-S24 model, Params: 21M
        """
        def __init__(self, **kwargs):
            layers = [4, 4, 12, 4]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, 
                mlp_ratios=mlp_ratios, downsamples=downsamples, 
                fork_feat=True,
                **kwargs)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s36_feat(PoolFormer):
        """
        PoolFormer-S36 model, Params: 31M
        """
        def __init__(self, **kwargs):
            layers = [6, 6, 18, 6]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, 
                mlp_ratios=mlp_ratios, downsamples=downsamples, 
                layer_scale_init_value=1e-6, 
                fork_feat=True,
                **kwargs)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_m36_feat(PoolFormer):
        """
        PoolFormer-S36 model, Params: 56M
        """
        def __init__(self, **kwargs):
            layers = [6, 6, 18, 6]
            embed_dims = [96, 192, 384, 768]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, 
                mlp_ratios=mlp_ratios, downsamples=downsamples, 
                layer_scale_init_value=1e-6, 
                fork_feat=True,
                **kwargs)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_m48_feat(PoolFormer):
        """
        PoolFormer-M48 model, Params: 73M
        """
        def __init__(self, **kwargs):
            layers = [8, 8, 24, 8]
            embed_dims = [96, 192, 384, 768]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, 
                mlp_ratios=mlp_ratios, downsamples=downsamples, 
                layer_scale_init_value=1e-6, 
                fork_feat=True,
                **kwargs)
            
if __name__ == "__main__":
    model = poolformer_eeda_m48().cuda()
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
    # Print the flops of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
    
    print("num_paras: ", params)
