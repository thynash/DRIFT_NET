from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class PDEDataConfig:
    dataset_name: str = "navier_stokes"
    image_size: int = 64
    in_channels: int = 1
    train_trajectories: int = 128
    val_trajectories: int = 32
    rollout_trajectories: int = 16
    trajectory_length: int = 12
    rollout_horizon: int = 50
    dt: float = 0.05
    viscosity: float = 2.0e-3
    diffusion_rate: float = 7.5e-2
    burgers_viscosity: float = 1.0e-2
    forcing_scale: float = 4.0e-2
    noise_std: float = 1.0e-3
    seed: int = 42

    @classmethod
    def from_dict(cls, config: dict) -> "PDEDataConfig":
        return cls(**config)


class SpectralGrid:
    """
    Cached Fourier grid on CPU for synthetic PDE generation.
    """

    def __init__(self, resolution: int) -> None:
        width_rfft = resolution // 2 + 1
        self.resolution = resolution
        self.width_rfft = width_rfft

        ky = 2.0 * math.pi * torch.fft.fftfreq(resolution, d=1.0 / resolution)
        kx = 2.0 * math.pi * torch.fft.rfftfreq(resolution, d=1.0 / resolution)
        grid_y = ky[:, None].expand(resolution, width_rfft)
        grid_x = kx[None, :].expand(resolution, width_rfft)

        self.kx = grid_x
        self.ky = grid_y
        self.k2 = (grid_x.square() + grid_y.square()).clamp_min(1.0)
        self.k2[0, 0] = 1.0

        dealias_x = grid_x.abs() <= (2.0 / 3.0) * grid_x.abs().max()
        dealias_y = grid_y.abs() <= (2.0 / 3.0) * grid_y.abs().max()
        self.dealias = (dealias_x & dealias_y).float()


class BasePDEGenerator:
    """
    Base class for synthetic pseudo-spectral PDE generators.
    """

    def __init__(self, config: PDEDataConfig, split_seed: int) -> None:
        self.config = config
        self.resolution = int(config.image_size)
        self.channels = int(config.in_channels)
        if self.channels != 1:
            raise ValueError("The provided synthetic generators currently assume single-channel scalar fields.")

        self.grid = SpectralGrid(self.resolution)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(split_seed)

    def sample_smooth_field(self, amplitude: float = 1.0, decay: float = 0.045) -> Tensor:
        width_rfft = self.grid.width_rfft
        radius = torch.sqrt(self.grid.kx.square() + self.grid.ky.square())
        envelope = torch.exp(-decay * radius.square())

        real = torch.randn((self.resolution, width_rfft), generator=self.generator)
        imag = torch.randn((self.resolution, width_rfft), generator=self.generator)
        coeffs = torch.complex(real, imag) * envelope
        coeffs[0, 0] = 0.0
        field = torch.fft.irfft2(coeffs, s=(self.resolution, self.resolution), norm="ortho")
        field = amplitude * field / field.std().clamp_min(1.0e-6)
        return field.unsqueeze(0)  # (1, H, W)

    def sample_forcing(self) -> Tensor:
        return self.config.forcing_scale * self.sample_smooth_field(amplitude=1.0, decay=0.02)

    def _stabilize_state(self, state: Tensor, max_abs_value: float = 10.0) -> Tensor:
        state = torch.nan_to_num(state, nan=0.0, posinf=max_abs_value, neginf=-max_abs_value)
        state = state - state.mean(dim=(-2, -1), keepdim=True)
        state = state / state.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        state = state.clamp(min=-max_abs_value, max=max_abs_value)
        return state

    def _diffusion_step(self, state: Tensor) -> Tensor:
        # state: (1, H, W)
        state_hat = torch.fft.rfft2(state, dim=(-2, -1), norm="ortho")
        decay = torch.exp(-self.config.diffusion_rate * self.config.dt * self.grid.k2)
        next_hat = state_hat * decay
        next_state = torch.fft.irfft2(next_hat, s=(self.resolution, self.resolution), dim=(-2, -1), norm="ortho")
        if self.config.noise_std > 0.0:
            noise = torch.randn(next_state.shape, generator=self.generator, dtype=next_state.dtype)
            next_state = next_state + self.config.noise_std * noise
        return self._stabilize_state(next_state)

    def _burgers_step(self, state: Tensor, forcing: Tensor) -> Tensor:
        # state/forcing: (1, H, W)
        state_hat = torch.fft.rfft2(state, dim=(-2, -1), norm="ortho")
        u2 = 0.5 * state.square()
        u2_hat = torch.fft.rfft2(u2, dim=(-2, -1), norm="ortho")
        nonlinear_hat = 1j * (self.grid.kx.unsqueeze(0) + self.grid.ky.unsqueeze(0)) * u2_hat
        nonlinear_hat = nonlinear_hat * self.grid.dealias.unsqueeze(0)

        forcing_hat = torch.fft.rfft2(forcing, dim=(-2, -1), norm="ortho")
        numerator = state_hat - self.config.dt * nonlinear_hat + self.config.dt * forcing_hat
        denominator = 1.0 + self.config.dt * self.config.burgers_viscosity * self.grid.k2.unsqueeze(0)
        next_hat = numerator / denominator
        next_state = torch.fft.irfft2(next_hat, s=(self.resolution, self.resolution), dim=(-2, -1), norm="ortho")
        return self._stabilize_state(next_state)

    def _navier_stokes_surrogate_step(self, state: Tensor, forcing: Tensor) -> Tensor:
        # state/forcing: (1, H, W) vorticity-like scalar field
        w_hat = torch.fft.rfft2(state, dim=(-2, -1), norm="ortho")
        psi_hat = -w_hat / self.grid.k2.unsqueeze(0)
        psi_hat[..., 0, 0] = 0.0

        velocity_x_hat = 1j * self.grid.ky.unsqueeze(0) * psi_hat
        velocity_y_hat = -1j * self.grid.kx.unsqueeze(0) * psi_hat

        velocity_x = torch.fft.irfft2(
            velocity_x_hat,
            s=(self.resolution, self.resolution),
            dim=(-2, -1),
            norm="ortho",
        )
        velocity_y = torch.fft.irfft2(
            velocity_y_hat,
            s=(self.resolution, self.resolution),
            dim=(-2, -1),
            norm="ortho",
        )

        dw_dx_hat = 1j * self.grid.kx.unsqueeze(0) * w_hat
        dw_dy_hat = 1j * self.grid.ky.unsqueeze(0) * w_hat
        dw_dx = torch.fft.irfft2(dw_dx_hat, s=(self.resolution, self.resolution), dim=(-2, -1), norm="ortho")
        dw_dy = torch.fft.irfft2(dw_dy_hat, s=(self.resolution, self.resolution), dim=(-2, -1), norm="ortho")

        advection = velocity_x * dw_dx + velocity_y * dw_dy
        advection_hat = torch.fft.rfft2(advection, dim=(-2, -1), norm="ortho") * self.grid.dealias.unsqueeze(0)
        forcing_hat = torch.fft.rfft2(forcing, dim=(-2, -1), norm="ortho")

        numerator = w_hat - self.config.dt * advection_hat + self.config.dt * forcing_hat
        denominator = 1.0 + self.config.dt * self.config.viscosity * self.grid.k2.unsqueeze(0)
        next_hat = numerator / denominator
        next_state = torch.fft.irfft2(next_hat, s=(self.resolution, self.resolution), dim=(-2, -1), norm="ortho")
        return self._stabilize_state(next_state)

    def step(self, state: Tensor, forcing: Tensor) -> Tensor:
        if self.config.dataset_name == "diffusion":
            return self._diffusion_step(state)
        if self.config.dataset_name == "burgers":
            return self._burgers_step(state, forcing)
        if self.config.dataset_name == "navier_stokes":
            return self._navier_stokes_surrogate_step(state, forcing)
        raise ValueError(f"Unknown dataset_name '{self.config.dataset_name}'.")

    def generate_trajectory(self, length: int) -> Tensor:
        """
        Returns:
            trajectory: (T, 1, H, W)
        """

        state = self.sample_smooth_field()
        forcing = self.sample_forcing()
        states: List[Tensor] = [state.clone()]
        for _ in range(length - 1):
            state = self.step(state, forcing)
            states.append(state.clone())
        return torch.stack(states, dim=0)


class SyntheticPDEPairsDataset(Dataset):
    """
    Dataset of one-step forecasting pairs ``(u_t, u_{t+1})``.
    """

    def __init__(self, generator: BasePDEGenerator, num_trajectories: int, trajectory_length: int) -> None:
        super().__init__()
        self.inputs: List[Tensor] = []
        self.targets: List[Tensor] = []

        for _ in range(num_trajectories):
            trajectory = generator.generate_trajectory(trajectory_length)  # (T, 1, H, W)
            for index in range(trajectory.shape[0] - 1):
                self.inputs.append(trajectory[index].float())
                self.targets.append(trajectory[index + 1].float())

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.inputs[index], self.targets[index]


class SyntheticPDERolloutDataset(Dataset):
    """
    Dataset of full trajectories for autoregressive rollout evaluation.

    Each item:
        trajectory: (T + 1, 1, H, W)
    """

    def __init__(self, generator: BasePDEGenerator, num_trajectories: int, rollout_horizon: int) -> None:
        super().__init__()
        self.trajectories: List[Tensor] = []
        for _ in range(num_trajectories):
            self.trajectories.append(generator.generate_trajectory(rollout_horizon + 1).float())

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> Tensor:
        return self.trajectories[index]


class DiffusionPairsDataset(SyntheticPDEPairsDataset):
    def __init__(self, config: PDEDataConfig, num_trajectories: int, trajectory_length: int, split_seed: int) -> None:
        diffusion_config = PDEDataConfig(**{**config.__dict__, "dataset_name": "diffusion"})
        super().__init__(BasePDEGenerator(diffusion_config, split_seed=split_seed), num_trajectories, trajectory_length)


class BurgersPairsDataset(SyntheticPDEPairsDataset):
    def __init__(self, config: PDEDataConfig, num_trajectories: int, trajectory_length: int, split_seed: int) -> None:
        burgers_config = PDEDataConfig(**{**config.__dict__, "dataset_name": "burgers"})
        super().__init__(BasePDEGenerator(burgers_config, split_seed=split_seed), num_trajectories, trajectory_length)


class NavierStokesPairsDataset(SyntheticPDEPairsDataset):
    def __init__(self, config: PDEDataConfig, num_trajectories: int, trajectory_length: int, split_seed: int) -> None:
        ns_config = PDEDataConfig(**{**config.__dict__, "dataset_name": "navier_stokes"})
        super().__init__(BasePDEGenerator(ns_config, split_seed=split_seed), num_trajectories, trajectory_length)


def build_datasets(config: PDEDataConfig) -> Tuple[SyntheticPDEPairsDataset, SyntheticPDEPairsDataset, SyntheticPDERolloutDataset]:
    """
    Construct train/validation one-step datasets and validation rollout dataset.
    """

    train_generator = BasePDEGenerator(config, split_seed=config.seed)
    val_generator = BasePDEGenerator(config, split_seed=config.seed + 1)
    rollout_generator = BasePDEGenerator(config, split_seed=config.seed + 2)

    train_dataset = SyntheticPDEPairsDataset(
        generator=train_generator,
        num_trajectories=config.train_trajectories,
        trajectory_length=config.trajectory_length,
    )
    val_dataset = SyntheticPDEPairsDataset(
        generator=val_generator,
        num_trajectories=config.val_trajectories,
        trajectory_length=config.trajectory_length,
    )
    rollout_dataset = SyntheticPDERolloutDataset(
        generator=rollout_generator,
        num_trajectories=config.rollout_trajectories,
        rollout_horizon=config.rollout_horizon,
    )
    return train_dataset, val_dataset, rollout_dataset


def describe_dataset(config: PDEDataConfig) -> dict:
    return {
        "dataset_name": config.dataset_name,
        "image_size": config.image_size,
        "train_trajectories": config.train_trajectories,
        "val_trajectories": config.val_trajectories,
        "rollout_trajectories": config.rollout_trajectories,
        "trajectory_length": config.trajectory_length,
        "rollout_horizon": config.rollout_horizon,
        "dt": config.dt,
    }
