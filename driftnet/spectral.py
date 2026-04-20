from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


def _inverse_sigmoid(value: float) -> float:
    value = min(max(value, 1.0e-4), 1.0 - 1.0e-4)
    return math.log(value / (1.0 - value))


@dataclass
class SpectralMixOutput:
    """
    Container returned by :class:`LowFrequencyMixer`.

    Shapes:
        spectrum:        (B, C, H, W_rfft) complex
        mixed_low:       (B, C, H, W_rfft) complex
        high_residual:   (B, C, H, W_rfft) complex
        low_mask:        (1, 1, H, W_rfft) real
        cutoff_height:   () real
        cutoff_width:    () real
    """

    spectrum: Tensor
    mixed_low: Tensor
    high_residual: Tensor
    low_mask: Tensor
    cutoff_height: Tensor
    cutoff_width: Tensor


@dataclass
class SpectralFusionOutput:
    """
    Container returned by :class:`SpectralFusion`.

    Shapes:
        spatial:         (B, C, H, W) real
        fused_spectrum:  (B, C, H, W_rfft) complex
        alpha_map:       (B, 1, H, W_rfft) real
        low_mask:        (1, 1, H, W_rfft) real
    """

    spatial: Tensor
    fused_spectrum: Tensor
    alpha_map: Tensor
    low_mask: Tensor


class LowFrequencyMixer(nn.Module):
    """
    Controlled low-frequency spectral mixing.

    The module:
    1. Applies ``rfft2`` to the input feature map.
    2. Constructs a learnable rectangular low-frequency mask.
    3. Applies complex channel mixing only on the low-frequency region.
    4. Preserves the complementary high-frequency residual unchanged.

    Input:
        x: (B, C, H, W) real

    Output:
        SpectralMixOutput
    """

    def __init__(
        self,
        channels: int,
        init_cutoff_height: float = 0.25,
        init_cutoff_width: float = 0.25,
        min_cutoff: float = 0.05,
        max_cutoff: float = 0.95,
        mask_temperature: float = 0.04,
        fft_norm: str = "ortho",
        enable_mixing: bool = True,
        mixing_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.min_cutoff = float(min_cutoff)
        self.max_cutoff = float(max_cutoff)
        self.mask_temperature = float(mask_temperature)
        self.fft_norm = fft_norm
        self.enable_mixing = enable_mixing
        self.mixing_residual_scale = float(mixing_residual_scale)

        self.cutoff_height_logit = nn.Parameter(torch.tensor(_inverse_sigmoid(init_cutoff_height)))
        self.cutoff_width_logit = nn.Parameter(torch.tensor(_inverse_sigmoid(init_cutoff_width)))

        self.register_buffer("identity_real", torch.eye(channels, dtype=torch.float32), persistent=False)
        self.weight_real_delta = nn.Parameter(torch.zeros(channels, channels, dtype=torch.float32))
        self.weight_imag_delta = nn.Parameter(torch.zeros(channels, channels, dtype=torch.float32))

    def _cutoff_ratio(self, logit: Tensor) -> Tensor:
        ratio = torch.sigmoid(logit)
        return self.min_cutoff + (self.max_cutoff - self.min_cutoff) * ratio

    def _build_rectangular_mask(
        self,
        height: int,
        width_rfft: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        cutoff_height = self._cutoff_ratio(self.cutoff_height_logit)
        cutoff_width = self._cutoff_ratio(self.cutoff_width_logit)

        full_width = max(1, 2 * (width_rfft - 1))
        freq_y = torch.fft.fftfreq(height, device=device).abs().to(dtype=dtype)
        freq_x = torch.fft.rfftfreq(full_width, device=device).to(dtype=dtype)

        freq_y = freq_y / freq_y.max().clamp_min(1.0e-6)
        freq_x = freq_x / freq_x.max().clamp_min(1.0e-6)

        mask_y = torch.sigmoid((cutoff_height - freq_y) / self.mask_temperature)
        mask_x = torch.sigmoid((cutoff_width - freq_x) / self.mask_temperature)
        low_mask = (mask_y[:, None] * mask_x[None, :]).unsqueeze(0).unsqueeze(0)
        return low_mask, cutoff_height, cutoff_width

    def forward(self, x: Tensor) -> SpectralMixOutput:
        # x: (B, C, H, W) real
        batch_size, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, received {channels}.")

        # spectrum: (B, C, H, W_rfft) complex
        spectrum = torch.fft.rfft2(x, dim=(-2, -1), norm=self.fft_norm)
        width_rfft = spectrum.shape[-1]

        low_mask, cutoff_height, cutoff_width = self._build_rectangular_mask(
            height=height,
            width_rfft=width_rfft,
            device=x.device,
            dtype=x.dtype,
        )

        # low_coeffs/high_residual: (B, C, H, W_rfft) complex
        low_coeffs = spectrum * low_mask
        high_residual = spectrum * (1.0 - low_mask)

        if self.enable_mixing:
            mixing_real = self.identity_real + self.mixing_residual_scale * torch.tanh(self.weight_real_delta)
            mixing_imag = self.mixing_residual_scale * torch.tanh(self.weight_imag_delta)
            mixing_matrix = torch.complex(mixing_real, mixing_imag)  # (C, C)
            mixed_low = torch.einsum("oc,bchw->bohw", mixing_matrix, low_coeffs)
        else:
            mixed_low = low_coeffs

        mixed_low = torch.nan_to_num(mixed_low, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        high_residual = torch.nan_to_num(high_residual, nan=0.0, posinf=1.0e4, neginf=-1.0e4)

        return SpectralMixOutput(
            spectrum=spectrum,
            mixed_low=mixed_low,
            high_residual=high_residual,
            low_mask=low_mask,
            cutoff_height=cutoff_height,
            cutoff_width=cutoff_width,
        )


class RadialBandGate(nn.Module):
    """
    Bandwise radial gating for Fourier-space fusion.

    The half-plane ``rfft`` spectrum is partitioned into ``J`` radial bands.
    Each band pools low- and high-spectrum magnitudes, then a tiny MLP predicts
    a scalar gate for the entire band.

    Inputs:
        low_spectrum:   (B, C, H, W_rfft) complex
        high_spectrum:  (B, C, H, W_rfft) complex

    Output:
        alpha_map:      (B, 1, H, W_rfft) real in [0, 1]
    """

    def __init__(
        self,
        num_bands: int = 8,
        hidden_dim: int = 32,
        eps: float = 1.0e-6,
        enable_gating: bool = True,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.eps = float(eps)
        self.enable_gating = enable_gating

        self.band_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.band_bias = nn.Parameter(torch.zeros(self.num_bands))
        self._cache: Dict[Tuple[int, int, str], Tuple[Tensor, Tensor, Tensor]] = {}

    def _cache_key(self, height: int, width_rfft: int, device: torch.device) -> Tuple[int, int, str]:
        return height, width_rfft, f"{device.type}:{device.index}"

    def _band_tensors(
        self,
        height: int,
        width_rfft: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        key = self._cache_key(height, width_rfft, device)
        if key in self._cache:
            band_masks, band_counts, band_ids = self._cache[key]
            return band_masks.to(dtype=dtype), band_counts.to(dtype=dtype), band_ids

        full_width = max(1, 2 * (width_rfft - 1))
        freq_y = torch.fft.fftfreq(height, device=device)
        freq_x = torch.fft.rfftfreq(full_width, device=device)

        grid_y = freq_y[:, None].expand(height, width_rfft)
        grid_x = freq_x[None, :].expand(height, width_rfft)
        radius = torch.sqrt(grid_x.square() + grid_y.square())
        radius = radius / radius.max().clamp_min(self.eps)

        edges = torch.linspace(0.0, 1.0, steps=self.num_bands + 1, device=device)
        band_ids = torch.bucketize(radius.reshape(-1), edges[1:-1], right=False).reshape(height, width_rfft)

        masks = torch.stack([(band_ids == band).float() for band in range(self.num_bands)], dim=0)
        counts = masks.sum(dim=(-2, -1)).clamp_min(1.0)
        self._cache[key] = (masks, counts, band_ids.long())
        return masks.to(dtype=dtype), counts.to(dtype=dtype), band_ids.long()

    def forward(self, low_spectrum: Tensor, high_spectrum: Tensor) -> Tensor:
        if low_spectrum.shape != high_spectrum.shape:
            raise ValueError("Low- and high-frequency inputs must have matching shapes.")

        batch_size, _, height, width_rfft = low_spectrum.shape
        band_masks, band_counts, band_ids = self._band_tensors(
            height=height,
            width_rfft=width_rfft,
            device=low_spectrum.device,
            dtype=low_spectrum.real.dtype,
        )

        if not self.enable_gating:
            return torch.zeros(
                batch_size,
                1,
                height,
                width_rfft,
                device=low_spectrum.device,
                dtype=low_spectrum.real.dtype,
            )

        low_mag = torch.log1p(low_spectrum.abs()).mean(dim=1)    # (B, H, W_rfft)
        high_mag = torch.log1p(high_spectrum.abs()).mean(dim=1)  # (B, H, W_rfft)

        low_flat = low_mag.reshape(batch_size, -1)
        high_flat = high_mag.reshape(batch_size, -1)
        masks_flat = band_masks.reshape(self.num_bands, -1)

        pooled_low_mean = torch.einsum("bn,jn->bj", low_flat, masks_flat) / band_counts[None, :]
        pooled_high_mean = torch.einsum("bn,jn->bj", high_flat, masks_flat) / band_counts[None, :]

        centered_low = (low_mag.unsqueeze(1) - pooled_low_mean[:, :, None, None]) * band_masks[None]
        centered_high = (high_mag.unsqueeze(1) - pooled_high_mean[:, :, None, None]) * band_masks[None]

        pooled_low_std = torch.sqrt(
            centered_low.square().sum(dim=(-2, -1)) / band_counts[None, :].clamp_min(self.eps)
        )
        pooled_high_std = torch.sqrt(
            centered_high.square().sum(dim=(-2, -1)) / band_counts[None, :].clamp_min(self.eps)
        )

        band_features = torch.stack(
            [
                pooled_low_mean,
                pooled_high_mean,
                pooled_low_std,
                pooled_high_std,
            ],
            dim=-1,
        )  # (B, J, 4)
        band_features = torch.nan_to_num(band_features, nan=0.0, posinf=50.0, neginf=-50.0)
        alpha_bands = torch.sigmoid(self.band_mlp(band_features).squeeze(-1) + self.band_bias[None, :])

        gather_index = band_ids.reshape(1, -1).expand(batch_size, -1)
        alpha_map = alpha_bands.gather(dim=1, index=gather_index).reshape(batch_size, 1, height, width_rfft)
        return alpha_map


class SpectralFusion(nn.Module):
    """
    Complete spectral branch used inside a DRIFT block.

    Fusion rule:
        Y_hat(k) = alpha(k) * V_low(k) + (1 - alpha(k)) * X_high(k)

    where:
        V_low   = low-frequency mixed spectrum
        X_high  = untouched high-frequency residual

    Input:
        x: (B, C, H, W) real

    Output:
        SpectralFusionOutput
    """

    def __init__(
        self,
        channels: int,
        num_bands: int = 8,
        gating_hidden_dim: int = 32,
        init_cutoff_height: float = 0.25,
        init_cutoff_width: float = 0.25,
        fft_norm: str = "ortho",
        enable_low_frequency_mixing: bool = True,
        enable_radial_gating: bool = True,
    ) -> None:
        super().__init__()
        self.fft_norm = fft_norm
        self.low_frequency_mixer = LowFrequencyMixer(
            channels=channels,
            init_cutoff_height=init_cutoff_height,
            init_cutoff_width=init_cutoff_width,
            fft_norm=fft_norm,
            enable_mixing=enable_low_frequency_mixing,
        )
        self.radial_gate = RadialBandGate(
            num_bands=num_bands,
            hidden_dim=gating_hidden_dim,
            enable_gating=enable_radial_gating,
        )
        self.enable_radial_gating = enable_radial_gating

    def forward(self, x: Tensor) -> SpectralFusionOutput:
        # x: (B, C, H, W) real
        _, _, height, width = x.shape
        mix = self.low_frequency_mixer(x)

        if self.enable_radial_gating:
            alpha_map = self.radial_gate(mix.mixed_low, mix.high_residual)  # (B, 1, H, W_rfft)
        else:
            alpha_map = mix.low_mask.expand(x.shape[0], -1, -1, -1)

        fused_spectrum = alpha_map * mix.mixed_low + (1.0 - alpha_map) * mix.high_residual
        fused_spectrum = torch.nan_to_num(fused_spectrum, nan=0.0, posinf=1.0e4, neginf=-1.0e4)

        # spatial: (B, C, H, W) real
        spatial = torch.fft.irfft2(fused_spectrum, s=(height, width), dim=(-2, -1), norm=self.fft_norm)
        spatial = torch.nan_to_num(spatial, nan=0.0, posinf=1.0e4, neginf=-1.0e4)

        return SpectralFusionOutput(
            spatial=spatial,
            fused_spectrum=fused_spectrum,
            alpha_map=alpha_map,
            low_mask=mix.low_mask,
        )
