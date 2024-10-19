import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=True, device=None):
        super().__init__()
        assert dim % heads == 0, "dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.mask = torch.tril(torch.ones(window_size, window_size)).to(
            device)  # mask for causality

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)  # [B*T/window_size,window_size,ndim]
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  #[B*T/window_size, self.num_heads, window_size, C // self.num_heads]

        # merge key padding and attention masks
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, T, T]

        # if self.causal:
        #     attn = attn.masked_fill_(self.mask == 0, float("-inf"))
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:  # reshape to the original size
            x = x.reshape(B_prev, T_prev, C_prev)
        return x
class CT_MSA(nn.Module):
    # Causal Temporal MSA
    def __init__(self,
                 dim,  # hidden dim
                 depth,  # the number of MSA in CT-MSA
                 heads,  # the number of heads
                 window_size,  # the size of local window
                 num_time,  # the number of time slot
                 dropout=0.,  # dropout rate
                 device=None,
                 causal=True,
                 pos=False):  # device, e.g., cuda
        super().__init__()
        self.pos = pos
        if pos==True:
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_time, dim))
        self.attn_ln = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim,
                                  heads=heads,
                                  window_size=window_size[i],
                                  dropout=dropout,
                                  device=device,
                                  causal=causal),
                PreNorm(dim, FeedForward(dim, dim*4, dropout=dropout))
            ]))

    def forward(self, x):
        # x: [b,t,dim]
        b,t,dim = x.shape
        if self.pos==True:
            x = x + self.pos_embedding  #自动广播机制
        x = self.attn_ln(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
# Pre Normalization in Transformer

class PreNorm(nn.Module):
    def __init__(self, dim, feedforward):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.feedforward = feedforward

    def forward(self, x, **kwargs):
        return self.feedforward(self.norm(x), **kwargs)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn([2,100,1280])
    model = CT_MSA(dim=1280,depth=3,heads=4,window_size=[5,10,20],num_time=100)
    model = model.to(device)