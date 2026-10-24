# LWGANet-derived components: Copyright (c) Microsoft Corporation; MIT License.
from __future__ import annotations

import math
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from timm.layers import DropPath, trunc_normal_
except ModuleNotFoundError:
    from timm.models.layers import DropPath, trunc_normal_

# The DRFD, attention, and stem components below are adapted from LWGANet.

try:
    import antialiased_cnns
except ImportError:
    class BlurPool(nn.Module):
        def __init__(self, channels: int, stride: int = 2):
            super().__init__()
            filt = torch.tensor([1.0, 2.0, 1.0])
            filt = filt[:, None] * filt[None, :]
            filt = filt / filt.sum()
            self.register_buffer("filt", filt[None, None].repeat(channels, 1, 1, 1))
            self.pad = nn.ReflectionPad2d(1)
            self.stride = stride
            self.groups = channels

        def forward(self, x: Tensor) -> Tensor:
            return F.conv2d(self.pad(x), self.filt, stride=self.stride, groups=self.groups)

    class antialiased_cnns:
        BlurPool = BlurPool


class DRFD(nn.Module):
    """Downsampling block used between hierarchical stages."""
    def __init__(self, dim: int, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Conv2d(dim, 2 * dim, 3, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(2 * dim, 2 * dim, 3, stride=2, padding=1, groups=2 * dim)
        self.act_c = act_layer()
        self.norm_c = norm_layer(2 * dim)
        self.max_m = nn.MaxPool2d(3, stride=2, padding=1)
        self.norm_m = norm_layer(2 * dim)
        self.fusion = nn.Conv2d(4 * dim, 2 * dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        conv = self.norm_c(self.act_c(self.conv_c(x)))
        pooled = self.norm_m(self.max_m(x))
        return self.fusion(torch.cat((conv, pooled), dim=1))


class PA(nn.Module):
    def __init__(self, dim: int, norm_layer, act_layer):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, 1, bias=False), norm_layer(4 * dim), act_layer(),
            nn.Conv2d(4 * dim, dim, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.proj(x).sigmoid()


class LA(nn.Module):
    def __init__(self, dim: int, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False), norm_layer(dim), act_layer()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class MRA(nn.Module):
    def __init__(self, channel: int, att_kernel: int, norm_layer):
        super().__init__()
        pad = att_kernel // 2
        self.channel = channel
        self.pool_local = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool_global = antialiased_cnns.BlurPool(channel, stride=3)
        self.h1 = nn.Conv2d(channel, channel, (att_kernel, 3), padding=(pad, 1), groups=channel, bias=False)
        self.v1 = nn.Conv2d(channel, channel, (3, att_kernel), padding=(1, pad), groups=channel, bias=False)
        self.h2 = nn.Conv2d(channel, channel, (att_kernel, 3), padding=(pad, 1), groups=channel, bias=False)
        self.v2 = nn.Conv2d(channel, channel, (3, att_kernel), padding=(1, pad), groups=channel, bias=False)
        self.norm = norm_layer(channel)

    @staticmethod
    def h_transform(x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = F.pad(x, (0, w)).reshape(b, c, -1)[..., :-w]
        return x.reshape(b, c, h, 2 * w - 1)

    @staticmethod
    def inv_h_transform(x: Tensor) -> Tensor:
        b, c, h, _ = x.shape
        x = F.pad(x.reshape(b, c, -1).contiguous(), (0, h))
        return x.reshape(b, c, h, 2 * h)[..., :h]

    @staticmethod
    def v_transform(x: Tensor) -> Tensor:
        return MRA.h_transform(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

    @staticmethod
    def inv_v_transform(x: Tensor) -> Tensor:
        return MRA.inv_h_transform(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.pool_global(self.pool_local(x))
        attn = self.h1(y) + self.v1(y)
        attn = attn + self.inv_h_transform(self.h2(self.h_transform(y)))
        attn = attn + self.inv_v_transform(self.v2(self.v_transform(y)))
        attn = self.norm(attn).sigmoid()
        return x[:, :self.channel] * F.interpolate(attn, size=x.shape[-2:], mode="nearest")


class GA12(nn.Module):
    def __init__(self, dim: int, act_layer):
        super().__init__()
        self.down = nn.MaxPool2d(2, 2, return_indices=True)
        self.up = nn.MaxUnpool2d(2, 2)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = act_layer()
        self.local = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.spatial = nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.fuse = nn.Conv2d(dim // 2, dim, 1)
        self.proj2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        y, indices = self.down(x)
        y = self.act(self.proj1(y))
        a, b = self.local(y), self.spatial(self.local(y))
        a, b = self.conv1(a), self.conv2(b)
        joined = torch.cat((a, b), dim=1)
        pooled = torch.cat((joined.mean(1, keepdim=True), joined.amax(1, keepdim=True)), dim=1)
        weights = self.squeeze(pooled).sigmoid()
        y = self.proj2(self.fuse(a * weights[:, 0:1] + b * weights[:, 1:2]))
        return self.up(y, indices=indices)


class GA(nn.Module):
    def __init__(self, dim: int, head_dim: int = 4, num_heads: int | None = None,
                 qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.,
                 proj_bias: bool = False, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads or max(dim // head_dim, 1)
        self.attention_dim = self.num_heads * head_dim
        self.qkv = nn.Linear(dim, 3 * self.attention_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1)
        qkv = self.qkv(tokens).reshape(b, h * w, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(b, h, w, self.attention_dim)
        return self.proj_drop(self.proj(y)).permute(0, 3, 1, 2)


class D_GA(nn.Module):
    def __init__(self, dim: int, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = GA(dim)
        self.down = nn.MaxPool2d(2, 2, return_indices=True)
        self.up = nn.MaxUnpool2d(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        y, indices = self.down(x)
        return self.up(self.norm(self.attn(y)), indices=indices)


class Stem(nn.Module):
    def __init__(self, in_chans: int, stem_dim: int, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, stem_dim, kernel_size=4, stride=4, bias=False)
        self.norm = norm_layer(stem_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class FrequencyBandGate(nn.Module):
    def __init__(self, dim: int, band: str, cutoff: float = 0.25, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU):
        super().__init__()
        if band not in {"high", "low"}:
            raise ValueError("band must be high or low")
        self.band = band
        self.cutoff = cutoff
        hidden = max(dim // 4, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            act_layer(),
            nn.Conv2d(hidden, dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            norm_layer(dim),
            act_layer(),
        )

    def _fft_band_energy(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        magnitude = spectrum.abs()
        fy = torch.fft.fftfreq(h, device=x.device).abs().view(1, 1, h, 1)
        fx = torch.fft.rfftfreq(w, device=x.device).abs().view(1, 1, 1, w // 2 + 1)
        radius = torch.sqrt(fx.square() + fy.square())
        if self.band == "low":
            mask = radius <= self.cutoff
        else:
            mask = radius > self.cutoff
        denom = mask.sum().clamp_min(1).to(magnitude.dtype)
        energy = (magnitude * mask.to(magnitude.dtype)).sum(dim=(2, 3), keepdim=True) / denom
        return energy.to(dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        band_feat = low if self.band == "low" else x - low
        gate = self.gate(self._fft_band_energy(x))
        return self.refine(band_feat) * gate


class DFGA(nn.Module):
    """Dual-Frequency Grouped Attention for high, low, and local channel groups."""
    def __init__(
        self,
        dim: int,
        cutoff: float,
        high_scale: float,
        low_scale: float,
        norm_layer,
        act_layer,
    ):
        super().__init__()
        self.high_attention = FrequencyBandGate(
            dim, "high", cutoff=cutoff, norm_layer=norm_layer, act_layer=act_layer
        )
        self.low_attention = FrequencyBandGate(
            dim, "low", cutoff=cutoff, norm_layer=norm_layer, act_layer=act_layer
        )
        self.local_attention = LA(dim, norm_layer, act_layer)
        self.high_scale = nn.Parameter(torch.tensor(high_scale, dtype=torch.float32))
        self.low_scale = nn.Parameter(torch.tensor(low_scale, dtype=torch.float32))

    def forward(self, x_high: Tensor, x_low: Tensor, x_local: Tensor):
        x_high = x_high + self.high_scale * self.high_attention(x_high)
        x_low = x_low + self.low_scale * self.low_attention(x_low)
        x_local = self.local_attention(x_local)
        return x_high, x_low, x_local


class SAGI(nn.Module):
    def __init__(self, dim: int, topk_ratio: float = 0.08, max_anchors: int = 24, qkv_bias: bool = False):
        super().__init__()
        self.topk_ratio = topk_ratio
        self.max_anchors = max_anchors
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.spatial_score = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        n = tokens.shape[1]
        k_num = max(1, min(self.max_anchors, int(n * self.topk_ratio)))

        score = self.spatial_score(x).flatten(2).squeeze(1)
        _, topk_idx = torch.topk(score, k=k_num, dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, c)
        topk_anchors = torch.gather(tokens, dim=1, index=gather_idx)
        mean_anchor = tokens.mean(dim=1, keepdim=True)
        anchors = torch.cat([mean_anchor, topk_anchors], dim=1)

        q = self.q(tokens)
        k = self.k(anchors)
        v = self.v(anchors)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(c)
        attn = attn.softmax(dim=-1)
        out = self.proj(torch.matmul(attn, v))
        return out.transpose(1, 2).reshape(b, c, h, w)


class FGSA_Block(nn.Module):
    def __init__(
        self,
        dim: int,
        stage: int,
        mlp_ratio: float,
        drop_path: float,
        act_layer,
        norm_layer,
        freq_cutoff: float = 0.25,
        topk_ratio: float = 0.08,
        high_scale: float = 0.10,
        low_scale: float = 0.30,
        global_scale: float = 0.60,
        att_kernel: int = 11,
        use_dfga: bool = True,
        use_sagi: bool = True,
    ):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 4
        self.use_dfga = use_dfga
        self.use_sagi = use_sagi
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.global_scale = nn.Parameter(torch.tensor(global_scale, dtype=torch.float32))

        if use_dfga:
            self.dfga = DFGA(
                self.dim_split,
                cutoff=freq_cutoff,
                high_scale=high_scale,
                low_scale=low_scale,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.base_pa = PA(self.dim_split, norm_layer, act_layer)
            self.base_la = LA(self.dim_split, norm_layer, act_layer)
            self.base_mra = MRA(self.dim_split, att_kernel, norm_layer)

        if use_sagi:
            self.sagi = SAGI(
                self.dim_split,
                topk_ratio=topk_ratio,
                max_anchors=12 if stage < 2 else 24,
            )
            self.global_norm = norm_layer(self.dim_split)
        elif stage == 2:
            self.base_global = D_GA(self.dim_split, norm_layer)
        elif stage == 3:
            self.base_global = GA(self.dim_split)
            self.global_norm = norm_layer(self.dim_split)
        else:
            self.base_global = GA12(self.dim_split, act_layer)
            self.global_norm = norm_layer(self.dim_split)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
        )
        self.norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x1, x2, x3, x4 = torch.split(x, [self.dim_split] * 4, dim=1)
        if self.use_dfga:
            x1, x2, x3 = self.dfga(x1, x2, x3)
        else:
            x1 = x1 + self.base_pa(x1)
            x2 = self.base_la(x2)
            x3 = self.base_mra(x3)
        if self.use_sagi:
            x4 = self.global_norm(x4 + self.global_scale * self.sagi(x4))
        elif self.stage == 2:
            x4 = x4 + self.base_global(x4)
        else:
            x4 = self.global_norm(x4 + self.base_global(x4))
        x_att = torch.cat((x1, x2, x3, x4), dim=1)
        return shortcut + self.norm(self.drop_path(self.mlp(x_att)))


class FGSA_Stage(nn.Module):
    def __init__(
        self,
        dim: int,
        stage: int,
        depth: int,
        mlp_ratio: float,
        drop_path: list[float],
        norm_layer,
        act_layer,
        freq_cutoff: float,
        topk_ratio: float,
        high_scale: float,
        low_scale: float,
        global_scale: float,
        att_kernel: int,
        use_dfga: bool,
        use_sagi: bool,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                FGSA_Block(
                    dim=dim,
                    stage=stage,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    freq_cutoff=freq_cutoff,
                    topk_ratio=topk_ratio,
                    high_scale=high_scale,
                    low_scale=low_scale,
                    global_scale=global_scale,
                    att_kernel=att_kernel,
                    use_dfga=use_dfga,
                    use_sagi=use_sagi,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class FGSANet(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        stem_dim: int = 32,
        depths: tuple[int, int, int, int] = (1, 2, 4, 2),
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.GELU,
        mlp_ratio: float = 2.0,
        stem_norm: bool = True,
        feature_dim: int = 1280,
        drop_path_rate: float = 0.0,
        freq_cutoff: float = 0.25,
        topk_ratio: float = 0.08,
        high_scale: float = 0.10,
        low_scale: float = 0.30,
        global_scale: float = 0.60,
        att_kernel: int = 11,
        use_dfga: bool = True,
        use_sagi: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.num_features = int(stem_dim * 2 ** (self.num_stages - 1))
        self.stem = Stem(in_chans=in_chans, stem_dim=stem_dim, norm_layer=norm_layer if stem_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        stages_list: List[nn.Module] = []
        for i_stage in range(self.num_stages):
            dim = int(stem_dim * 2**i_stage)
            stages_list.append(
                FGSA_Stage(
                    dim=dim,
                    stage=i_stage,
                    depth=depths[i_stage],
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[sum(depths[:i_stage]) : sum(depths[: i_stage + 1])],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    freq_cutoff=freq_cutoff,
                    topk_ratio=topk_ratio,
                    high_scale=high_scale,
                    low_scale=low_scale,
                    global_scale=global_scale,
                    att_kernel=att_kernel,
                    use_dfga=use_dfga,
                    use_sagi=use_sagi,
                )
            )
            if i_stage < self.num_stages - 1:
                stages_list.append(DRFD(dim=dim, norm_layer=norm_layer, act_layer=act_layer))

        self.stages = nn.Sequential(*stages_list)
        self.avgpool_pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
            act_layer(),
        )
        self.head = nn.Linear(feature_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        return self.stages(x)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> Tensor:
        x = self.forward_features(x)
        x = self.avgpool_pre_head(x)
        x = torch.flatten(x, 1)
        return self.head(x)


