import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

try:
    from .modules import IRB, PatchEmbed, Attention, Hydra_Embedding_Dimensional_Attention, Embedding_Dimensional_Attention, Efficient_Linear_Embedding_Dimensional_Attention, Cross_Covariance_Attention, Efficient_Channel_Attention, Linear_Embedding_Dimensional_Attention
except:
    from modules import IRB, PatchEmbed, Attention, Hydra_Embedding_Dimensional_Attention, Embedding_Dimensional_Attention, Efficient_Linear_Embedding_Dimensional_Attention, Cross_Covariance_Attention, Efficient_Channel_Attention, Linear_Embedding_Dimensional_Attention

module_dict = {
    "EEDA": Efficient_Linear_Embedding_Dimensional_Attention,
    "HEDA": Hydra_Embedding_Dimensional_Attention,
    "LEDA": Linear_Embedding_Dimensional_Attention,
    "EDA": Embedding_Dimensional_Attention,
    "XCAtt": Cross_Covariance_Attention,
    "ECA": Efficient_Channel_Attention,
    "MLP": IRB
}

class Block(nn.Module):

    def __init__(self, dim, num_heads, num_patches, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ff_module="MLP"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        if ff_module == "MLP":
            self.ff = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=drop, ksize=3)
        else:
            self.ff = module_dict[ff_module](dim, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale, eda_num_heads=8,
                    attn_drop=attn_drop, proj_drop=drop, num_patches=num_patches, mlp_hidden_dim=int(dim * mlp_ratio), drop=0.)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ff(self.norm2(x)))

        return x


class Pure_Vision_Transformer(nn.Module):
    def __init__(self, attention_list=[], img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=768,
                 num_heads=12, mlp_ratios=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=12, **kwargs): #
        super().__init__()
        
        if not len(attention_list) == depths:
            raise ValueError("attention_list length must be equal to depths")
        self.num_classes = num_classes
        self.depths = depths

        self.embed_dims = embed_dims

        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, kernel_size=3, in_chans=in_chans,
                                       embed_dim=embed_dims, overlap=False)  
        self.pos_embeddings = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
        
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        cur = 0

        ksize = 3

        self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_patches=num_patches, ff_module=attention_list[i])
            for i in range(depths)])


        
        # classification head
        # self.head = nn.Linear(embed_dims[2], num_classes) if num_classes > 0 else nn.Identity()
        # Multi classification heads
        self.head = nn.Linear(embed_dims, num_classes)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.apply(self._init_weights)

        #print(self)

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
        self.head = nn.Linear(self.embed_dims, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed(x)
        x += self.pos_embeddings
        
        for idx, blk in enumerate(self.block):
            x = blk(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = torch.mean(x, dim=1)
        x = self.head(x)        
        return x
    
@register_model
def pure_vit(pretrained=False, **kwargs):
    model = Pure_Vision_Transformer(**kwargs)
    return model
    

if __name__ == "__main__":
    model = pure_vit(False, img_size=224, num_classes=1000, patch_size=14, attention_list=[
        "EEDA",
        "EEDA",
        "EEDA",
        "EEDA",
        "EEDA",
        "EEDA",
        "EEDA",
        "EEDA",
        "EEDA",
        "EEDA",
        "EEDA",
        "EEDA"]).cuda()
    print(model)
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