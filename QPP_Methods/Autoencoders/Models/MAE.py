#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.nn as nn
import numpy as np


# In[5]:


class PatchEmbedding(nn.Module):
    
    def __init__(self, img_size, patch_size, num_input_channels, embedding_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_input_channels = num_input_channels
        self.embedding_dim = embedding_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
         num_input_channels,
         embedding_dim,
         kernel_size = patch_size,
         stride = patch_size
        )
        
        
    def forward(self, x):
        
        # Input form
        # N_samplex X Image_size X Image_size
        
        # N_samples x Embedding_dimension x Sqrt(N_Patches) x Sqrt(N_Patches)
        x = self.proj(x)
        
        # N_samples X Embedding_dimension x N_Patches
        x = x.flatten(2)
        
        # N_samplex X N_Patches X Embedding_dimension
        x = x.transpose(1,2)
        
        return x


# In[45]:


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(3, B, N, self.num_heads, C // self.num_heads)
        q, k, v = qkv[0],qkv[1],qkv[2]
        
        attn = torch.einsum('bqhd,bkhd->bhqk',[q,k]) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)
        
        x = torch.einsum('bhqk,bvhd->bqhd',[attn, v])
        x = torch.flatten(x,-2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# In[ ]:


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# In[4]:


class LayerScale(nn.Module):
    def __init__(self, dim, init_values = 1e-5, inplace=False):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x):
        return x * self.gamma


# In[5]:


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size).astype(float)
    grid_w = np.arange(grid_size).astype(float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed



def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2).astype(float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# In[6]:


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'    


# In[7]:


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()
        
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x
        
        


# In[2]:


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4.0, qkv_bias=True, proj_drop=0., attn_drop=0., drop_path = 0.):
        
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps = 1e-6)
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
        self.ls1   = LayerScale(dim)
        self.ls2   = LayerScale(dim)
        self.attn = MultiHeadAttention(dim, num_heads = num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = proj_drop)
        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features = dim , hidden_features = hidden_features, out_features = dim)
        
        
    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# In[2]:


class MAETransformer(nn.Module):
    def __init__(self, img_size = 512, patch_size = 16, num_input_channels = 3,
                 encoder_embedding_dim = 768, decoder_embed_dim = 512, depth = 12,decoder_depth = 8, num_heads = 12, decoder_num_heads= 16, mlp_ratio = 4.0, qkv_bias = True,
                 device = 'cpu',
                 p=0., attn_p = 0.):
        
        super().__init__()
        
        
        # Masked Encoder
        
        self.patch_embed = PatchEmbedding(img_size = img_size, patch_size = patch_size,
                                      num_input_channels = num_input_channels,
                                      embedding_dim = encoder_embedding_dim).to(device)
        
        self.cls_token = nn.Parameter(torch.zeros(1,1,encoder_embedding_dim)).to(device)
        self.pos_embed = nn.Parameter(torch.zeros(1,1+self.patch_embed.num_patches, encoder_embedding_dim), requires_grad = False).to(device) # Feeding fixed positional encodings
        
        
        self.encoder_blocks = nn.ModuleList(
        [
            TransformerBlock(dim = encoder_embedding_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias)
            for _ in range(depth)
        ]).to(device)
        self.encoder_norm = nn.LayerNorm(encoder_embedding_dim, eps=1e-6).to(device)
        
        
        # Masked Decoder
        
        
        self.decoder_embed = nn.Linear(encoder_embedding_dim, decoder_embed_dim).to(device)
        self.mask_token = nn.Parameter(torch.zeros(1,1,decoder_embed_dim)).to(device)
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1+self.patch_embed.num_patches, decoder_embed_dim), requires_grad = False).to(device) # Feeding fixed positional encodings
        self.decoder_blocks = nn.ModuleList([
        TransformerBlock(dim = decoder_embed_dim, num_heads = decoder_num_heads, mlp_ratio = mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(decoder_depth)
        ]).to(device)
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim).to(device)
        self.decoder_lin  = nn.Linear(decoder_embed_dim , patch_size **2 * num_input_channels, bias = True).to(device)
        
        
        self.initialize_weights()
        
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)     
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)  
            
    def patchify(self, x):
        
        p = self.patch_embed.patch_size
        
        assert x.shape[2] == x.shape[3] and x.shape[2] % p == 0
        
        h = w = x.shape[2] // p
        x = x.reshape((x.shape[0], 3, h , p, w , p))
        x = torch.einsum('nchpwq->nhwpqc',x )
        x = x.reshape((x.shape[0], h * w, p**2 *3))
        
        return x
    
    
    def unpatchify(self, x):
        
        p = self.patch_embed.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        
        x = x.reshape((x.shape[0], h, w, p, p,3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape((x.shape[0], 3, h*p, w*p))
        
        return x
    
    def random_masking(self, x , mask_ratio):
        
        N,L,D = x.shape
        
        len_keep = int ( L *(1 - mask_ratio))
        
        noise = torch.rand(N,L,device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim = 1)
        ids_restore = torch.argsort(ids_shuffle, dim = 1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index = ids_keep.unsqueeze(-1).repeat(1,1,D))
        
        mask = torch.ones([N,L], device = x.device)
        mask [:, :len_keep] = 0
        mask = torch.gather(mask, dim = 1, index = ids_restore)
        
        return x_masked, mask, ids_restore
        
        
    def forward_enc(self, x, mask_ratio):
        n_samples = x.shape[0]
        
        
        x = self.patch_embed(x)
        
        x = x + self.pos_embed[:, 1:, :]
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        
        cls_token = self.cls_token + self.pos_embed[:,:1,:]
        cls_tokens = cls_token.expand(n_samples, -1 , -1)
        x = torch.cat((cls_tokens,x),dim=1)
        
        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_dec(self, x , ids_restore):
        
        x  = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:,1:,:], mask_tokens],dim = 1)
        x_ = torch.gather(x_, dim = 1, index = ids_restore.unsqueeze(-1).repeat(1,1,x.shape[2])) # unshuffle
        x = torch.cat([x[:,:1,:],x_], dim = 1)
        
        x = x + self.decoder_pos_embed
        
        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.decoder_norm(x)
        
        x = self.decoder_lin(x)
        
        x = x[:,1:,:]
        
        return x
    
    def forward_loss(self, x , pred, mask):
        target = self.patchify(x)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim = -1)
        
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, x , mask_ratio = 0.75):
        
        latent, mask, ids_restore = self.forward_enc(x, mask_ratio)
        pred = self.forward_dec(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

