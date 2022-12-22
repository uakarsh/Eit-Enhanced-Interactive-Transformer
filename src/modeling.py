import torch
import torch.nn as nn
from einops import rearrange

## Pre-Normalization

class PreNorm(nn.Module):
    def __init__(self, dim : int, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttn(nn.Module):
    def __init__(self, dim : int, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, feat, **kwargs):
        return self.fn(self.norm(feat), **kwargs)

## Feed forward output

class FeedForward(nn.Module):
    def __init__(self, dim : int, hidden_dim : int, dropout : float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


## Multi-head attention

class ConvolutionOp(nn.Module):

  def __init__(self, in_channels : int = 144, out_channels : int = 12,
               groups : int = 12, h_isi : int = 96, h_csi : int = 48): 
    super().__init__()

    self.conv_inner_subspace_1 = nn.Conv2d(in_channels = in_channels, 
                 out_channels = h_isi, 
                 kernel_size = 1, groups = groups)
    self.conv_inner_subspace_2 = nn.Conv2d(in_channels = h_isi, 
                 out_channels = out_channels, 
                 kernel_size = 1, groups = groups)
    

    self.conv_cross_subspace_1 = nn.Conv2d(in_channels = out_channels, 
                 out_channels = h_csi, 
                 kernel_size = 1, groups = groups)
    self.conv_cross_subspace_2 = nn.Conv2d(in_channels = h_csi, 
                 out_channels = out_channels, 
                 kernel_size = 1, groups = groups)

  def forward(self, attn_map):
    attn_map = rearrange(attn_map, 'head b t d -> b head t d')
    s_dot = self.conv_inner_subspace_2(nn.ReLU()(self.conv_inner_subspace_1(attn_map)))
    s_double_dot = self.conv_cross_subspace_2(nn.ReLU()(self.conv_cross_subspace_1(s_dot)))
    s_double_dot = rearrange(s_double_dot, 'b head t d -> head b t d')
    return s_double_dot


class EfficientConvOp(nn.Module):

  def __init__(self, in_channels : int = 144, out_channels : int = 12,
               groups : int = 12, common_channels : int = 24):
    super().__init__()

    self.conv_inner_subspace_1 = nn.Conv2d(in_channels = in_channels, 
                 out_channels = common_channels, 
                 kernel_size = 1, groups = groups)
    
    self.conv_cross_subspace_1 = nn.Conv2d(in_channels = common_channels, 
                 out_channels = out_channels, 
                 kernel_size = 1, groups = groups)

  def forward(self, attn_map):
    attn_map = rearrange(attn_map, 'head b t d -> b head t d')
    s_dot = nn.ReLU()(self.conv_inner_subspace_1(attn_map))
    s_double_dot = self.conv_cross_subspace_1(s_dot)
    s_double_dot = rearrange(s_double_dot, 'b head t d -> head b t d')
    return s_double_dot

class EMHA(nn.Module): ## Enhanced MHA
    def __init__(self, embed_dim : int = 768, n_heads : int = 12, dropout : float = 0., use_efficient : bool = False):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # qkv embeddings
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.scale = embed_dim ** 0.5

        if use_efficient:
          self.conv_op = EfficientConvOp(in_channels = n_heads ** 2, out_channels = n_heads,
               groups = n_heads, common_channels = 2 * n_heads)

        else:
          self.conv_op = ConvolutionOp(in_channels = n_heads ** 2, out_channels = n_heads,
               groups = n_heads, h_isi = 8 * n_heads, h_csi = 4 * n_heads)

    def forward(self, x):

      ## q, k and v vector
      query_x = rearrange(self.fc_q(x), 'b t (head k) -> head b t k', head = self.n_heads)
      key_x = rearrange(self.fc_k(x), 'b t (head k) -> head b t k', head = self.n_heads)
      value_x = rearrange(self.fc_v(x), 'b t (head k) -> head b t k', head = self.n_heads)

      attention_map = []
      for i in range(1, (self.n_heads ** 2) + 1):
        curr_q = int((i - 1) / self.n_heads)  ## Arguments should be b/w 0 to self.n_heads ** 2 - 1
        curr_k = int((i - 1) % self.n_heads)

        curr_attn = torch.einsum('blk,btk->blt', query_x[curr_q], key_x[curr_k]) / self.scale
        attention_map.append(curr_attn)

      attention_map = torch.stack(attention_map, axis = 0)  ## self.n_heads ** 2, b, l, t
      attention_map = self.conv_op(attention_map)

      context_vector = torch.einsum('hblt,hbtv->hblv', attention_map, value_x)
      context_vector = rearrange(context_vector, 'head b t d -> b t (head d)')
      return context_vector

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        for _ in range(config['num_hidden_layers']):
            encoder_block = nn.ModuleList([
                PreNormAttn(config['hidden_size'],
                            EMHA(embed_dim = config['hidden_size'],
                                                     n_heads = config['num_attention_heads'],
                                                     dropout = config['hidden_dropout_prob'],
                                                     use_efficient = config["use_efficient"]
                                                     )
                            ),
                PreNorm(config['hidden_size'],
                        FeedForward(config['hidden_size'],
                                    config['hidden_size'] * config['intermediate_ff_size_factor'],
                                    dropout=config['hidden_dropout_prob']))
            ])
            self.layers.append(encoder_block)

    def forward(self,feat):
        for attn, ff in self.layers:
            skip = feat
            x = attn(feat) + skip
            x = ff(x) + x
            feat = x
        return feat


if __name__ == "__main__":

  config = {
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "intermediate_ff_size_factor": 4,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "use_efficient" : True
  }

  encoder = Encoder(config)
  sample_vect = torch.rand(1, 512, 768)
  out_vec = encoder(sample_vect)
  print(out_vec.shape)