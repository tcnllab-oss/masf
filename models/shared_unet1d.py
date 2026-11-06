import math
from functools import partial

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange

# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out=None, circular_padding=False):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.Conv1d(dim, default(dim_out, dim), 3, padding=1, padding_mode='circular' if circular_padding else 'zeros'))

def Downsample(dim, dim_out=None, circular_padding=False):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1, padding_mode='circular' if circular_padding else 'zeros')

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds
class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        x = x * 1000
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block(Module):
    def __init__(self, dim, dim_out, dropout=0., circular_padding=False):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1, padding_mode='circular' if circular_padding else 'zeros')
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0., circular_padding=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout=dropout, circular_padding=circular_padding)
        self.block2 = Block(dim_out, dim_out, circular_padding=circular_padding)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        return self.to_out(out)

class Attention(Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


class DualHeadUNet1D(Module):
    def __init__(self, dim, init_dim, out_dim_1, out_dim_2, dim_mults, channels, dropout=0., sinusoidal_pos_emb_theta=10000, attn_dim_head=32, attn_heads=4, circular_padding=False):
        super().__init__()
        padding_mode = "circular" if circular_padding else "zeros"
        self.init_conv = nn.Conv1d(channels, init_dim, 7, padding=3, padding_mode=padding_mode)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout, circular_padding=circular_padding)

        self.downs = ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down = nn.Conv1d(dim_in, dim_out, 3, padding=1, padding_mode=padding_mode) if ind >= len(in_out) - 1 else Downsample(dim_in, dim_out, circular_padding)
            self.downs.append(ModuleList([block(dim_in, dim_in), block(dim_in, dim_in), Residual(PreNorm(dim_in, LinearAttention(dim_in))), down]))

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block(mid_dim, mid_dim)

        self.ups_1 = self._make_ups(in_out, block, circular_padding, padding_mode)
        self.ups_2 = self._make_ups(in_out, block, circular_padding, padding_mode)
        self.final_res_block_1, self.final_res_block_2 = block(init_dim * 2, init_dim), block(init_dim * 2, init_dim)
        self.final_conv_1, self.final_conv_2 = nn.Conv1d(init_dim, out_dim_1, 1), nn.Conv1d(init_dim, out_dim_2, 1)

    @staticmethod
    def _make_ups(in_out, block, circular_padding, padding_mode):
        ups = ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            up = nn.Conv1d(dim_out, dim_in, 3, padding=1, padding_mode=padding_mode) if ind == len(in_out) - 1 else Upsample(dim_out, dim_in, circular_padding)
            ups.append(ModuleList([block(dim_out + dim_in, dim_out), block(dim_out + dim_in, dim_out), Residual(PreNorm(dim_out, LinearAttention(dim_out))), up]))
        return ups

    @staticmethod
    def _decode(x, skips, residual, time_emb, ups, final_res_block, final_conv):
        skips = list(skips)
        for block1, block2, attn, upsample in ups:
            x = block1(torch.cat((x, skips.pop()), dim=1), time_emb)
            x = block2(torch.cat((x, skips.pop()), dim=1), time_emb)
            x = upsample(attn(x))
        return final_conv(final_res_block(torch.cat((x, residual), dim=1), time_emb))

    def forward(self, x, time):
        x = self.init_conv(x)
        residual, time_emb, skips = x.clone(), self.time_mlp(time), []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            skips.append(x)
            x = attn(block2(x, time_emb))
            skips.append(x)
            x = downsample(x)
        x = self.mid_block2(self.mid_attn(self.mid_block1(x, time_emb)), time_emb)
        y1 = self._decode(x, skips, residual, time_emb, self.ups_1, self.final_res_block_1, self.final_conv_1)
        y2 = self._decode(x, skips, residual, time_emb, self.ups_2, self.final_res_block_2, self.final_conv_2)
        return y1, y2
