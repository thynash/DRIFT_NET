from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .spectral import SpectralFusion


class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm for ``(B, C, H, W)`` tensors.
    """

    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
        mask = mask.floor()
        return x * mask / keep_prob


class ConvNeXtLocalBranch(nn.Module):
    """
    ConvNeXt-style local branch.

    Structure:
        depthwise 3x3 conv -> LayerNorm -> pointwise expand -> GELU -> pointwise project

    Input:
        x: (B, C, H, W)

    Output:
        y: (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        layer_scale_init_value: float = 1.0e-6,
    ) -> None:
        super().__init__()
        hidden_channels = channels * expansion

        self.depthwise_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.norm = nn.LayerNorm(channels, eps=1.0e-6)
        self.pointwise_expand = nn.Linear(channels, hidden_channels)
        self.activation = nn.GELU()
        self.pointwise_project = nn.Linear(hidden_channels, channels)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        y = self.depthwise_conv(x)         # (B, C, H, W)
        y = y.permute(0, 2, 3, 1)          # (B, H, W, C)
        y = self.norm(y)                   # (B, H, W, C)
        y = self.pointwise_expand(y)       # (B, H, W, expansion*C)
        y = self.activation(y)             # (B, H, W, expansion*C)
        y = self.pointwise_project(y)      # (B, H, W, C)
        y = y * self.layer_scale.view(1, 1, 1, -1)
        y = y.permute(0, 3, 1, 2)          # (B, C, H, W)
        return y


class DRIFTBlock(nn.Module):
    """
    One DRIFT block with dual local/spectral branches.

    Forward:
        local = local_branch(x)
        spectral = spectral_branch(norm(x))
        fused = local + spectral
        y = x + fused

    Input:
        x: (B, C, H, W)

    Output:
        y: (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        spectral_bands: int = 8,
        spectral_gate_hidden_dim: int = 32,
        init_cutoff_height: float = 0.25,
        init_cutoff_width: float = 0.25,
        fft_norm: str = "ortho",
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1.0e-6,
        enable_local_branch: bool = True,
        enable_spectral_branch: bool = True,
        enable_low_frequency_mixing: bool = True,
        enable_radial_gating: bool = True,
    ) -> None:
        super().__init__()
        self.enable_local_branch = enable_local_branch
        self.enable_spectral_branch = enable_spectral_branch

        self.local_branch: Optional[ConvNeXtLocalBranch]
        if enable_local_branch:
            self.local_branch = ConvNeXtLocalBranch(
                channels=channels,
                expansion=expansion,
                layer_scale_init_value=layer_scale_init_value,
            )
        else:
            self.local_branch = None

        self.spectral_norm: Optional[LayerNorm2d]
        self.spectral_branch: Optional[SpectralFusion]
        if enable_spectral_branch:
            self.spectral_norm = LayerNorm2d(channels)
            self.spectral_branch = SpectralFusion(
                channels=channels,
                num_bands=spectral_bands,
                gating_hidden_dim=spectral_gate_hidden_dim,
                init_cutoff_height=init_cutoff_height,
                init_cutoff_width=init_cutoff_width,
                fft_norm=fft_norm,
                enable_low_frequency_mixing=enable_low_frequency_mixing,
                enable_radial_gating=enable_radial_gating,
            )
        else:
            self.spectral_norm = None
            self.spectral_branch = None

        self.residual_scale = nn.Parameter(torch.ones(channels))
        self.drop_path = DropPath(drop_prob=drop_path)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        fused = torch.zeros_like(x)

        if self.local_branch is not None:
            fused = fused + self.local_branch(x)  # (B, C, H, W)

        if self.spectral_branch is not None and self.spectral_norm is not None:
            spectral_input = self.spectral_norm(x)                     # (B, C, H, W)
            spectral_out = self.spectral_branch(spectral_input).spatial  # (B, C, H, W)
            fused = fused + spectral_out

        fused = fused * self.residual_scale.view(1, -1, 1, 1)
        return x + self.drop_path(fused)
