from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


def build_radial_frequency_weight(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    power: float = 2.0,
    bias: float = 1.0,
) -> Tensor:
    """
    Build a monotonically increasing radial weight on the ``rfft2`` half-plane.

    Output:
        weight: (H, W_rfft)
    """

    freq_y = torch.fft.fftfreq(height, device=device)
    freq_x = torch.fft.rfftfreq(width, device=device)
    grid_y = freq_y[:, None].expand(height, freq_x.numel())
    grid_x = freq_x[None, :].expand(height, freq_x.numel())
    radius = torch.sqrt(grid_x.square() + grid_y.square())
    radius = radius / radius.max().clamp_min(1.0e-6)
    weight = bias + radius.pow(power)
    return weight.to(dtype=dtype)


class RelativeL1Loss(nn.Module):
    """
    Relative L1 loss:

        ||prediction - target||_1 / (||target||_1 + eps)

    Inputs:
        prediction: (B, C, H, W)
        target:     (B, C, H, W)
    """

    def __init__(self, eps: float = 1.0e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        error = (prediction - target).abs().flatten(start_dim=1).sum(dim=1)
        reference = target.abs().flatten(start_dim=1).sum(dim=1).clamp_min(self.eps)
        value = error / reference
        if self.reduction == "mean":
            return value.mean()
        if self.reduction == "sum":
            return value.sum()
        if self.reduction == "none":
            return value
        raise ValueError(f"Unsupported reduction '{self.reduction}'.")


class FrequencyWeightedLoss(nn.Module):
    """
    Frequency-domain error penalty that emphasizes higher frequencies.

    Steps:
        1. Compute ``rfft2`` of the spatial prediction error.
        2. Weight Fourier magnitudes radially, with larger weights at larger radii.
        3. Average across channels and samples.
    """

    def __init__(self, power: float = 2.0, bias: float = 1.0, fft_norm: str = "ortho") -> None:
        super().__init__()
        self.power = float(power)
        self.bias = float(bias)
        self.fft_norm = fft_norm

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        error = prediction - target                                  # (B, C, H, W)
        error_hat = torch.fft.rfft2(error, dim=(-2, -1), norm=self.fft_norm)
        weight = build_radial_frequency_weight(
            height=prediction.shape[-2],
            width=prediction.shape[-1],
            device=prediction.device,
            dtype=prediction.dtype,
            power=self.power,
            bias=self.bias,
        )                                                            # (H, W_rfft)
        weighted_error = error_hat.abs() * weight.unsqueeze(0).unsqueeze(0)
        return weighted_error.mean()


@dataclass
class LossBreakdown:
    total: Tensor
    base: Tensor
    frequency: Tensor


class DriftLoss(nn.Module):
    """
    Combined training loss:

        L = RelativeL1 + lambda_frequency * FrequencyWeightedLoss
    """

    def __init__(
        self,
        lambda_frequency: float = 0.1,
        frequency_power: float = 2.0,
        frequency_bias: float = 1.0,
        fft_norm: str = "ortho",
    ) -> None:
        super().__init__()
        self.lambda_frequency = float(lambda_frequency)
        self.base_loss = RelativeL1Loss(reduction="mean")
        self.frequency_loss = FrequencyWeightedLoss(
            power=frequency_power,
            bias=frequency_bias,
            fft_norm=fft_norm,
        )

    def forward(self, prediction: Tensor, target: Tensor) -> LossBreakdown:
        base = self.base_loss(prediction, target)
        frequency = self.frequency_loss(prediction, target)
        total = base + self.lambda_frequency * frequency
        return LossBreakdown(total=total, base=base, frequency=frequency)

    def as_dict(self, breakdown: LossBreakdown) -> Dict[str, float]:
        return {
            "loss_total": float(breakdown.total.detach().item()),
            "loss_base": float(breakdown.base.detach().item()),
            "loss_frequency": float(breakdown.frequency.detach().item()),
        }
