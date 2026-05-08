import copy
import torch as th
import torch.nn as nn
from abc import abstractmethod
from .unet import ResBlock, AttentionBlock, Downsample, Upsample, TimestepEmbedSequential
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,

)
# from .fp16_util import convert_module_to_f16, convert_module_to_f32

import math
import torch as th
import torch.nn as nn


class DualHeadUNetModel(nn.Module):
    """
    Shared encoder + middle, separate decoders for:
      - head1: m1 = E[X0 | Xt]
      - head2: m2 = E[h(X0) | Xt]
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels_1,
        out_channels_2,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads


        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)

        # -------------------------
        # Shared encoder
        # -------------------------
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # -------------------------
        # Shared middle
        # -------------------------
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # -------------------------
        # Separate decoders
        # -------------------------
        self.output_blocks_1, final_ch_1 = self._make_output_blocks(
            ch=ch,
            input_block_chans=input_block_chans.copy(),
            time_embed_dim=time_embed_dim,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            num_heads_upsample=num_heads_upsample,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            ds_init=ds,
        )

        self.output_blocks_2, final_ch_2 = self._make_output_blocks(
            ch=ch,
            input_block_chans=input_block_chans.copy(),
            time_embed_dim=time_embed_dim,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            num_heads_upsample=num_heads_upsample,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            ds_init=ds,
        )

        self.out_1 = nn.Sequential(
            normalization(final_ch_1),
            nn.SiLU(),
            zero_module(conv_nd(dims, final_ch_1, out_channels_1, 3, padding=1)),
        )
        self.out_2 = nn.Sequential(
            normalization(final_ch_2),
            nn.SiLU(),
            zero_module(conv_nd(dims, final_ch_2, out_channels_2, 3, padding=1)),
        )

    def _make_output_blocks(
        self,
        ch,
        input_block_chans,
        time_embed_dim,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        dropout,
        channel_mult,
        conv_resample,
        dims,
        use_checkpoint,
        num_heads_upsample,
        num_head_channels,
        use_scale_shift_norm,
        resblock_updown,
        use_new_attention_order,
        ds_init,
    ):
        output_blocks = nn.ModuleList([])
        ds = ds_init
        cur_ch = ch

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        cur_ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                cur_ch = int(model_channels * mult)

                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            cur_ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )

                if level and i == num_res_blocks:
                    out_ch = cur_ch
                    layers.append(
                        ResBlock(
                            cur_ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(cur_ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2

                output_blocks.append(TimestepEmbedSequential(*layers))

        return output_blocks, cur_ch

    def _decode(self, h, hs, emb, output_blocks, out_head, x_dtype):
        hs_local = list(hs)
        for module in output_blocks:
            h = th.cat([h, hs_local.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x_dtype)
        return out_head(h)

    def forward(self, x, timesteps=None):
        hs = []

        if timesteps is not None:
            # timesteps = 1000 * timesteps
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        else:
            emb = None

        h = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        y1 = self._decode(h, hs, emb, self.output_blocks_1, self.out_1, x.dtype)
        y2 = self._decode(h, hs, emb, self.output_blocks_2, self.out_2, x.dtype)

        return y1, y2