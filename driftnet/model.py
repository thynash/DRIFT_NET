from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvNeXtLocalBranch, DRIFTBlock, DropPath, LayerNorm2d


@dataclass
class DriftNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    widths: Tuple[int, int, int, int] = (64, 128, 256, 512)
    blocks_per_level: int = 2
    expansion: int = 4
    stem_kernel_size: int = 3
    spectral_bands: int = 8
    spectral_gate_hidden_dim: int = 32
    init_cutoff_height: float = 0.25
    init_cutoff_width: float = 0.25
    fft_norm: str = "ortho"
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1.0e-6
    enable_local_branch: bool = True
    enable_spectral_branch: bool = True
    enable_low_frequency_mixing: bool = True
    enable_radial_gating: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> "DriftNetConfig":
        payload = dict(config)
        if "widths" in payload:
            payload["widths"] = tuple(payload["widths"])
        return cls(**payload)


@dataclass
class DriftOperatorNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    hidden_channels: int = 128
    depth: int = 8
    expansion: int = 4
    spectral_bands: int = 8
    spectral_gate_hidden_dim: int = 32
    init_cutoff_height: float = 0.25
    init_cutoff_width: float = 0.25
    fft_norm: str = "ortho"
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1.0e-6
    enable_local_branch: bool = True
    enable_spectral_branch: bool = True
    enable_low_frequency_mixing: bool = True
    enable_radial_gating: bool = True


@dataclass
class ScOTConfig:
    in_channels: int = 1
    out_channels: int = 1
    widths: Tuple[int, int, int, int] = (64, 128, 256, 512)
    blocks_per_level: int = 2
    num_heads: Tuple[int, int, int, int] = (4, 4, 8, 8)
    window_size: int = 8
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.05
    convnext_expansion: int = 4


class PatchEmbeddingStem(nn.Module):
    """
    Shallow patch embedding stem.

    Input:
        x: (B, C_in, H, W)

    Output:
        y: (B, C_out, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)   # (B, C_out, H, W)
        x = self.norm(x)   # (B, C_out, H, W)
        return x


class DownsampleBlock(nn.Module):
    """
    Stride-2 downsampling convolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_channels)
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.proj(x)
        return x


class UpsampleBlock(nn.Module):
    """
    Bilinear upsampling followed by a 3x3 projection convolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: Tensor, target_size: Tuple[int, int]) -> Tensor:
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.proj(x)
        return x


class DecoderFuse(nn.Module):
    """
    Fuse a decoder tensor with its encoder skip tensor.

    Inputs:
        x:    (B, C_dec, H, W)
        skip: (B, C_skip, H, W)
    """

    def __init__(self, decoder_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            LayerNorm2d(decoder_channels + skip_channels),
            nn.Conv2d(decoder_channels + skip_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x


class DriftStage(nn.Module):
    """
    Stack of DRIFT blocks at a single resolution.
    """

    def __init__(self, channels: int, config: DriftNetConfig, drop_path_rates: Sequence[float]) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                DRIFTBlock(
                    channels=channels,
                    expansion=config.expansion,
                    spectral_bands=config.spectral_bands,
                    spectral_gate_hidden_dim=config.spectral_gate_hidden_dim,
                    init_cutoff_height=config.init_cutoff_height,
                    init_cutoff_width=config.init_cutoff_width,
                    fft_norm=config.fft_norm,
                    drop_path=drop_path_rates[index],
                    layer_scale_init_value=config.layer_scale_init_value,
                    enable_local_branch=config.enable_local_branch,
                    enable_spectral_branch=config.enable_spectral_branch,
                    enable_low_frequency_mixing=config.enable_low_frequency_mixing,
                    enable_radial_gating=config.enable_radial_gating,
                )
                for index in range(len(drop_path_rates))
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class DRIFTNet(nn.Module):
    """
    U-Net style DRIFT-NET with four encoder-decoder levels.

    This implementation is intentionally written in an explicit layer-by-layer
    style so each DRIFT block in the encoder and decoder can be inspected,
    modified, or instrumented independently.
    """

    def __init__(self, config: DriftNetConfig) -> None:
        super().__init__()
        self.config = config

        if len(config.widths) != 4:
            raise ValueError("DRIFTNet requires exactly four widths for the 4-level hierarchy.")

        widths = list(config.widths)
        total_blocks = (len(widths) + len(widths) - 1) * config.blocks_per_level
        drop_path_rates = torch.linspace(0.0, config.drop_path_rate, steps=max(total_blocks, 1)).tolist()
        rate_offset = 0

        self.stem = PatchEmbeddingStem(
            in_channels=config.in_channels,
            out_channels=widths[0],
            kernel_size=config.stem_kernel_size,
        )

        self.encoder_blocks = nn.ModuleList()
        for width in widths:
            block_list = nn.ModuleList()
            stage_rates = drop_path_rates[rate_offset : rate_offset + config.blocks_per_level]
            rate_offset += config.blocks_per_level
            for block_index in range(config.blocks_per_level):
                block_list.append(
                    DRIFTBlock(
                        channels=width,
                        expansion=config.expansion,
                        spectral_bands=config.spectral_bands,
                        spectral_gate_hidden_dim=config.spectral_gate_hidden_dim,
                        init_cutoff_height=config.init_cutoff_height,
                        init_cutoff_width=config.init_cutoff_width,
                        fft_norm=config.fft_norm,
                        drop_path=stage_rates[block_index],
                        layer_scale_init_value=config.layer_scale_init_value,
                        enable_local_branch=config.enable_local_branch,
                        enable_spectral_branch=config.enable_spectral_branch,
                        enable_low_frequency_mixing=config.enable_low_frequency_mixing,
                        enable_radial_gating=config.enable_radial_gating,
                    )
                )
            self.encoder_blocks.append(block_list)

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock(widths[0], widths[1]),
                DownsampleBlock(widths[1], widths[2]),
                DownsampleBlock(widths[2], widths[3]),
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock(widths[3], widths[2]),
                UpsampleBlock(widths[2], widths[1]),
                UpsampleBlock(widths[1], widths[0]),
            ]
        )
        self.decoder_fusions = nn.ModuleList(
            [
                DecoderFuse(widths[2], widths[2], widths[2]),
                DecoderFuse(widths[1], widths[1], widths[1]),
                DecoderFuse(widths[0], widths[0], widths[0]),
            ]
        )

        self.decoder_blocks = nn.ModuleList()
        for width in (widths[2], widths[1], widths[0]):
            block_list = nn.ModuleList()
            stage_rates = drop_path_rates[rate_offset : rate_offset + config.blocks_per_level]
            rate_offset += config.blocks_per_level
            for block_index in range(config.blocks_per_level):
                block_list.append(
                    DRIFTBlock(
                        channels=width,
                        expansion=config.expansion,
                        spectral_bands=config.spectral_bands,
                        spectral_gate_hidden_dim=config.spectral_gate_hidden_dim,
                        init_cutoff_height=config.init_cutoff_height,
                        init_cutoff_width=config.init_cutoff_width,
                        fft_norm=config.fft_norm,
                        drop_path=stage_rates[block_index],
                        layer_scale_init_value=config.layer_scale_init_value,
                        enable_local_branch=config.enable_local_branch,
                        enable_spectral_branch=config.enable_spectral_branch,
                        enable_low_frequency_mixing=config.enable_low_frequency_mixing,
                        enable_radial_gating=config.enable_radial_gating,
                    )
                )
            self.decoder_blocks.append(block_list)

        self.head = nn.Sequential(
            LayerNorm2d(widths[0]),
            nn.Conv2d(widths[0], widths[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(widths[0], config.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def _run_block_list(self, x: Tensor, block_list: nn.ModuleList) -> Tensor:
        for block in block_list:
            x = block(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        skip_connections: List[Tensor] = []

        x0 = self.stem(x)          # (B, 64, H, W)
        e0 = self._run_block_list(x0, self.encoder_blocks[0])
        skip_connections.append(e0)

        x1 = self.downsamples[0](e0)   # (B, 128, H/2, W/2)
        e1 = self._run_block_list(x1, self.encoder_blocks[1])
        skip_connections.append(e1)

        x2 = self.downsamples[1](e1)   # (B, 256, H/4, W/4)
        e2 = self._run_block_list(x2, self.encoder_blocks[2])
        skip_connections.append(e2)

        x3 = self.downsamples[2](e2)   # (B, 512, H/8, W/8)
        bottleneck = self._run_block_list(x3, self.encoder_blocks[3])

        d2 = self.upsamples[0](bottleneck, target_size=skip_connections[2].shape[-2:])
        d2 = self.decoder_fusions[0](d2, skip_connections[2])
        d2 = self._run_block_list(d2, self.decoder_blocks[0])

        d1 = self.upsamples[1](d2, target_size=skip_connections[1].shape[-2:])
        d1 = self.decoder_fusions[1](d1, skip_connections[1])
        d1 = self._run_block_list(d1, self.decoder_blocks[1])

        d0 = self.upsamples[2](d1, target_size=skip_connections[0].shape[-2:])
        d0 = self.decoder_fusions[2](d0, skip_connections[0])
        d0 = self._run_block_list(d0, self.decoder_blocks[2])

        return self.head(d0)       # (B, C_out, H, W)


class DRIFTOperatorNet(nn.Module):
    """
    Single-resolution DRIFT variant built purely from stacked operator layers.

    This removes the encoder-decoder hierarchy and applies DRIFT blocks on one
    fixed-resolution latent grid.
    """

    def __init__(self, config: DriftOperatorNetConfig) -> None:
        super().__init__()
        self.config = config

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, config.hidden_channels, kernel_size=3, padding=1, bias=True),
            LayerNorm2d(config.hidden_channels),
        )

        drop_path_rates = torch.linspace(0.0, config.drop_path_rate, steps=max(config.depth, 1)).tolist()
        self.layers = nn.Sequential(
            *[
                DRIFTBlock(
                    channels=config.hidden_channels,
                    expansion=config.expansion,
                    spectral_bands=config.spectral_bands,
                    spectral_gate_hidden_dim=config.spectral_gate_hidden_dim,
                    init_cutoff_height=config.init_cutoff_height,
                    init_cutoff_width=config.init_cutoff_width,
                    fft_norm=config.fft_norm,
                    drop_path=drop_path_rates[index],
                    layer_scale_init_value=config.layer_scale_init_value,
                    enable_local_branch=config.enable_local_branch,
                    enable_spectral_branch=config.enable_spectral_branch,
                    enable_low_frequency_mixing=config.enable_low_frequency_mixing,
                    enable_radial_gating=config.enable_radial_gating,
                )
                for index in range(config.depth)
            ]
        )
        self.head = nn.Sequential(
            LayerNorm2d(config.hidden_channels),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(config.hidden_channels, config.out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)      # (B, hidden, H, W)
        x = self.layers(x)    # (B, hidden, H, W)
        x = self.head(x)      # (B, C_out, H, W)
        return x


def _window_partition(x: Tensor, window_size: int) -> Tensor:
    # x: (B, H, W, C)
    batch_size, height, width, channels = x.shape
    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, channels)
    return windows


def _window_reverse(windows: Tensor, window_size: int, height: int, width: int, batch_size: int) -> Tensor:
    channels = windows.shape[-1]
    x = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, channels)
    return x


class WindowAttention(nn.Module):
    """
    Multi-head self-attention applied inside local windows.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # x: (num_windows*B, window_size*window_size, C)
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, num_windows*B, heads, tokens, head_dim)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query * self.scale) @ key.transpose(-2, -1)  # (num_windows*B, heads, tokens, tokens)
        if attention_mask is not None:
            num_windows = attention_mask.shape[0]
            attention = attention.view(-1, num_windows, self.num_heads, tokens, tokens)
            attention = attention + attention_mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(batch_windows, self.num_heads, tokens, tokens)

        attention = attention.softmax(dim=-1)
        out = attention @ value
        out = out.transpose(1, 2).reshape(batch_windows, tokens, channels)
        out = self.proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SwinConvNeXtBlock(nn.Module):
    """
    scOT-style block:
        shifted window attention + MLP + ConvNeXt local residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        drop_path: float,
        convnext_expansion: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim, eps=1.0e-6)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads, window_size=window_size)
        self.drop_path1 = DropPath(drop_prob=drop_path)

        self.norm2 = nn.LayerNorm(dim, eps=1.0e-6)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio))
        self.drop_path2 = DropPath(drop_prob=drop_path)

        self.convnext = ConvNeXtLocalBranch(
            channels=dim,
            expansion=convnext_expansion,
            layer_scale_init_value=1.0e-6,
        )
        self.drop_path3 = DropPath(drop_prob=drop_path)

    def _attention_mask(self, padded_height: int, padded_width: int, device: torch.device) -> Tensor | None:
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, padded_height, padded_width, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        region_id = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = region_id
                region_id += 1

        mask_windows = _window_partition(img_mask, self.window_size).squeeze(-1)  # (num_windows, tokens)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
        attention_mask = attention_mask.masked_fill(attention_mask == 0, 0.0)
        return attention_mask

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        batch_size, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {channels}.")

        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size

        shortcut = x
        y = x.permute(0, 2, 3, 1)               # (B, H, W, C)
        y = self.norm1(y)
        if pad_h > 0 or pad_w > 0:
            y = F.pad(y, (0, 0, 0, pad_w, 0, pad_h))

        padded_height, padded_width = y.shape[1], y.shape[2]
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = _window_partition(y, self.window_size)                # (num_windows*B, ws*ws, C)
        attention_mask = self._attention_mask(padded_height, padded_width, x.device)
        attended = self.attn(windows, attention_mask=attention_mask)    # (num_windows*B, ws*ws, C)
        y = _window_reverse(attended, self.window_size, padded_height, padded_width, batch_size)

        if self.shift_size > 0:
            y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_h > 0 or pad_w > 0:
            y = y[:, :height, :width, :]
        y = y.permute(0, 3, 1, 2)                                       # (B, C, H, W)
        x = shortcut + self.drop_path1(y)

        shortcut = x
        y = x.permute(0, 2, 3, 1)
        y = self.norm2(y)
        y = self.mlp(y)
        y = y.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path2(y)

        x = x + self.drop_path3(self.convnext(x))
        return x


class ScOTStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        convnext_expansion: int,
        drop_path_rates: Sequence[float],
    ) -> None:
        super().__init__()
        blocks = []
        for index in range(depth):
            shift_size = 0 if index % 2 == 0 else window_size // 2
            blocks.append(
                SwinConvNeXtBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rates[index],
                    convnext_expansion=convnext_expansion,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class ScOTBaseline(nn.Module):
    """
    scOT-inspired hierarchical baseline.

    Public descriptions of scOT describe a U-Net style hierarchy combining
    Swin-style window attention with ConvNeXt residual blocks. This class
    implements that comparison backbone for controlled experiments.
    """

    def __init__(self, config: ScOTConfig) -> None:
        super().__init__()
        if len(config.widths) != 4 or len(config.num_heads) != 4:
            raise ValueError("ScOTBaseline expects four stage widths and four stage head counts.")

        self.config = config
        widths = list(config.widths)

        total_blocks = (len(widths) + len(widths) - 1) * config.blocks_per_level
        drop_path_rates = torch.linspace(0.0, config.drop_path_rate, steps=max(total_blocks, 1)).tolist()
        rate_offset = 0

        self.stem = PatchEmbeddingStem(config.in_channels, widths[0], kernel_size=3)

        self.encoder_stages = nn.ModuleList()
        for width, heads in zip(widths, config.num_heads):
            stage_rates = drop_path_rates[rate_offset : rate_offset + config.blocks_per_level]
            rate_offset += config.blocks_per_level
            self.encoder_stages.append(
                ScOTStage(
                    dim=width,
                    depth=config.blocks_per_level,
                    num_heads=heads,
                    window_size=config.window_size,
                    mlp_ratio=config.mlp_ratio,
                    convnext_expansion=config.convnext_expansion,
                    drop_path_rates=stage_rates,
                )
            )

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock(widths[0], widths[1]),
                DownsampleBlock(widths[1], widths[2]),
                DownsampleBlock(widths[2], widths[3]),
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock(widths[3], widths[2]),
                UpsampleBlock(widths[2], widths[1]),
                UpsampleBlock(widths[1], widths[0]),
            ]
        )
        self.decoder_fusions = nn.ModuleList(
            [
                DecoderFuse(widths[2], widths[2], widths[2]),
                DecoderFuse(widths[1], widths[1], widths[1]),
                DecoderFuse(widths[0], widths[0], widths[0]),
            ]
        )

        self.decoder_stages = nn.ModuleList()
        for width, heads in zip((widths[2], widths[1], widths[0]), (config.num_heads[2], config.num_heads[1], config.num_heads[0])):
            stage_rates = drop_path_rates[rate_offset : rate_offset + config.blocks_per_level]
            rate_offset += config.blocks_per_level
            self.decoder_stages.append(
                ScOTStage(
                    dim=width,
                    depth=config.blocks_per_level,
                    num_heads=heads,
                    window_size=config.window_size,
                    mlp_ratio=config.mlp_ratio,
                    convnext_expansion=config.convnext_expansion,
                    drop_path_rates=stage_rates,
                )
            )

        self.head = nn.Sequential(
            LayerNorm2d(widths[0]),
            nn.Conv2d(widths[0], widths[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(widths[0], config.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        skips: List[Tensor] = []

        x0 = self.stem(x)             # (B, 64, H, W)
        e0 = self.encoder_stages[0](x0)
        skips.append(e0)

        x1 = self.downsamples[0](e0)  # (B, 128, H/2, W/2)
        e1 = self.encoder_stages[1](x1)
        skips.append(e1)

        x2 = self.downsamples[1](e1)  # (B, 256, H/4, W/4)
        e2 = self.encoder_stages[2](x2)
        skips.append(e2)

        x3 = self.downsamples[2](e2)  # (B, 512, H/8, W/8)
        bottleneck = self.encoder_stages[3](x3)

        d2 = self.upsamples[0](bottleneck, target_size=skips[2].shape[-2:])
        d2 = self.decoder_fusions[0](d2, skips[2])
        d2 = self.decoder_stages[0](d2)

        d1 = self.upsamples[1](d2, target_size=skips[1].shape[-2:])
        d1 = self.decoder_fusions[1](d1, skips[1])
        d1 = self.decoder_stages[1](d1)

        d0 = self.upsamples[2](d1, target_size=skips[0].shape[-2:])
        d0 = self.decoder_fusions[2](d0, skips[0])
        d0 = self.decoder_stages[2](d0)

        return self.head(d0)          # (B, C_out, H, W)
