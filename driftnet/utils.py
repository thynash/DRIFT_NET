from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


try:
    import yaml
except ImportError:  # pragma: no cover - runtime dependency
    yaml = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - runtime dependency
    plt = None


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_output_dirs() -> Dict[str, Path]:
    root = repo_root()
    return {
        "results": root / "results",
        "plots": root / "plots",
        "tables": root / "tables",
    }


def ensure_output_dirs() -> Dict[str, Path]:
    paths = default_output_dirs()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (paths["results"] / "checkpoints").mkdir(parents=True, exist_ok=True)
    return paths


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (parameter for parameter in params if parameter.requires_grad)
    return sum(parameter.numel() for parameter in params)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to load config.yaml. Install with `pip install pyyaml`.")
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must deserialize to a dictionary.")
    return config


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


@torch.no_grad()
def rollout_autoregressive(model: nn.Module, initial_state: Tensor, horizon: int, device: torch.device) -> Tensor:
    """
    Inputs:
        initial_state: (B, C, H, W)

    Returns:
        rollout: (B, horizon, C, H, W)
    """

    was_training = model.training
    model.eval()

    state = initial_state.to(device)
    predictions: List[Tensor] = []
    for _ in range(horizon):
        state = model(state)
        predictions.append(state.detach().cpu())

    model.train(was_training)
    return torch.stack(predictions, dim=1)


def relative_l1(prediction: Tensor, target: Tensor, eps: float = 1.0e-6) -> Tensor:
    numerator = (prediction - target).abs().flatten(start_dim=1).sum(dim=1)
    denominator = target.abs().flatten(start_dim=1).sum(dim=1).clamp_min(eps)
    return numerator / denominator


def mse(prediction: Tensor, target: Tensor) -> Tensor:
    return ((prediction - target) ** 2).flatten(start_dim=1).mean(dim=1)


def throughput_benchmark(
    model: nn.Module,
    input_shape: Sequence[int],
    device: torch.device,
    warmup_steps: int = 10,
    benchmark_steps: int = 50,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    example = torch.randn(*input_shape, device=device)

    for _ in range(warmup_steps):
        _ = model(example)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(benchmark_steps):
        _ = model(example)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    total_samples = input_shape[0] * benchmark_steps
    model.train(was_training)
    return {
        "seconds_total": elapsed,
        "seconds_per_batch": elapsed / benchmark_steps,
        "samples_per_second": total_samples / max(elapsed, 1.0e-8),
    }


def benchmark_train_step(
    model: nn.Module,
    criterion: nn.Module,
    input_shape: Sequence[int],
    device: torch.device,
    warmup_steps: int = 5,
    benchmark_steps: int = 20,
) -> Dict[str, float]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    inputs = torch.randn(*input_shape, device=device)
    targets = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3], device=device)

    def _step() -> None:
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss_output = criterion(predictions, targets)
        if hasattr(loss_output, "total"):
            loss = loss_output.total
        else:
            loss = loss_output
        loss.backward()
        optimizer.step()

    for _ in range(warmup_steps):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(benchmark_steps):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    return {
        "train_step_seconds": elapsed / benchmark_steps,
    }


def peak_memory_megabytes(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))


def plot_series(
    x_values: Sequence[float],
    y_series: Dict[str, Sequence[float]],
    title: str,
    x_label: str,
    y_label: str,
    path: str | Path,
) -> Optional[Path]:
    if plt is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for label, values in y_series.items():
        plt.plot(x_values, values, label=label, linewidth=2.0)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_training_history(history: Sequence[Dict[str, Any]], path: str | Path) -> Optional[Path]:
    if not history:
        return None
    epochs = [row["epoch"] for row in history]
    series = {
        "train_total": [row["train_loss_total"] for row in history],
        "val_total": [row["val_loss_total"] for row in history],
        "train_rel_l1": [row["train_rel_l1"] for row in history],
        "val_rel_l1": [row["val_rel_l1"] for row in history],
    }
    return plot_series(
        x_values=epochs,
        y_series=series,
        title="Training Curves",
        x_label="Epoch",
        y_label="Loss / Relative L1",
        path=path,
    )
