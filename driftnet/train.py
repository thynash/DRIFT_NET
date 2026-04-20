from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .datasets import PDEDataConfig, build_datasets
from .evaluate import evaluate_one_step, evaluate_rollout
from .losses import DriftLoss
from .model import DRIFTNet, DriftNetConfig
from .utils import (
    ensure_output_dirs,
    load_yaml_config,
    plot_training_history,
    save_checkpoint,
    save_csv,
    save_json,
    seed_everything,
)


@dataclass
class OptimConfig:
    run_name: str = "driftnet_default"
    batch_size: int = 8
    epochs: int = 25
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 0
    log_every: int = 10
    save_every: int = 5

    @classmethod
    def from_dict(cls, config: dict) -> "OptimConfig":
        return cls(**config)


@dataclass
class LossConfig:
    lambda_frequency: float = 0.1
    frequency_power: float = 2.0
    frequency_bias: float = 1.0

    @classmethod
    def from_dict(cls, config: dict) -> "LossConfig":
        return cls(**config)


def build_dataloaders(data_config: PDEDataConfig, optim_config: OptimConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, rollout_dataset = build_datasets(data_config)

    loader_kwargs = {
        "batch_size": optim_config.batch_size,
        "num_workers": optim_config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    rollout_loader = DataLoader(rollout_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    return train_loader, val_loader, rollout_loader


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def create_model(model_config: DriftNetConfig) -> DRIFTNet:
    return DRIFTNet(model_config)


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _uses_fft_spectral_modules(model: nn.Module) -> bool:
    spectral_module_names = {"LowFrequencyMixer", "SpectralFusion"}
    for module in model.modules():
        if module.__class__.__name__ in spectral_module_names:
            return True
    return False


def _is_finite_tensor(x: Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: DriftLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip_norm: Optional[float] = None,
    log_every: int = 10,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_base = 0.0
    total_freq = 0.0
    total_rel_l1 = 0.0
    total_mse = 0.0
    total_examples = 0

    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.shape[0]

        if not _is_finite_tensor(inputs) or not _is_finite_tensor(targets):
            print(f"warning: skipping non-finite batch at step={step}")
            continue

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device=device, enabled=scaler is not None and scaler.is_enabled()):
            predictions = model(inputs)
            loss_breakdown = criterion(predictions, targets)
            loss = loss_breakdown.total

        if not _is_finite_tensor(predictions):
            print(f"warning: non-finite predictions at step={step}; skipping optimizer update")
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            continue
        if not torch.isfinite(loss):
            print(f"warning: non-finite loss at step={step}; skipping optimizer update")
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            continue

        if is_train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients_finite = True
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        gradients_finite = False
                        break
                if not gradients_finite:
                    print(f"warning: non-finite gradients at step={step}; skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gradients_finite = True
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        gradients_finite = False
                        break
                if not gradients_finite:
                    print(f"warning: non-finite gradients at step={step}; skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        with torch.no_grad():
            rel_l1 = ((predictions - targets).abs().flatten(start_dim=1).sum(dim=1) /
                      targets.abs().flatten(start_dim=1).sum(dim=1).clamp_min(1.0e-6)).mean()
            batch_mse = ((predictions - targets) ** 2).flatten(start_dim=1).mean(dim=1).mean()

        total_loss += float(loss_breakdown.total.detach().item()) * batch_size
        total_base += float(loss_breakdown.base.detach().item()) * batch_size
        total_freq += float(loss_breakdown.frequency.detach().item()) * batch_size
        total_rel_l1 += float(rel_l1.detach().item()) * batch_size
        total_mse += float(batch_mse.detach().item()) * batch_size
        total_examples += batch_size

        if is_train and (step % log_every == 0 or step == len(loader)):
            print(
                f"step={step:04d}/{len(loader):04d} "
                f"loss={total_loss / total_examples:.6f} "
                f"base={total_base / total_examples:.6f} "
                f"freq={total_freq / total_examples:.6f} "
                f"rel_l1={total_rel_l1 / total_examples:.6f}"
            )

    return {
        "loss_total": total_loss / max(total_examples, 1),
        "loss_base": total_base / max(total_examples, 1),
        "loss_frequency": total_freq / max(total_examples, 1),
        "rel_l1": total_rel_l1 / max(total_examples, 1),
        "mse": total_mse / max(total_examples, 1),
    }


def prepare_training_components(config: Dict[str, Any], device: torch.device) -> Tuple[DRIFTNet, DriftLoss, DataLoader, DataLoader, DataLoader, OptimConfig]:
    model_config = DriftNetConfig.from_dict(config["model"])
    data_config = PDEDataConfig.from_dict(config["data"])
    optim_config = OptimConfig.from_dict(config["training"])
    loss_config = LossConfig.from_dict(config["loss"])

    model = create_model(model_config).to(device)
    criterion = DriftLoss(
        lambda_frequency=loss_config.lambda_frequency,
        frequency_power=loss_config.frequency_power,
        frequency_bias=loss_config.frequency_bias,
        fft_norm=model_config.fft_norm,
    )
    train_loader, val_loader, rollout_loader = build_dataloaders(data_config, optim_config)
    return model, criterion, train_loader, val_loader, rollout_loader, optim_config


def train_model(
    config: Dict[str, Any],
    device: torch.device,
    model: Optional[nn.Module] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    outputs = ensure_output_dirs()

    model_config = DriftNetConfig.from_dict(config["model"])
    data_config = PDEDataConfig.from_dict(config["data"])
    optim_config = OptimConfig.from_dict(config["training"])
    loss_config = LossConfig.from_dict(config["loss"])
    if run_name is not None:
        optim_config.run_name = run_name

    criterion = DriftLoss(
        lambda_frequency=loss_config.lambda_frequency,
        frequency_power=loss_config.frequency_power,
        frequency_bias=loss_config.frequency_bias,
        fft_norm=model_config.fft_norm,
    )
    train_loader, val_loader, rollout_loader = build_dataloaders(data_config, optim_config)
    model = create_model(model_config) if model is None else model
    model = model.to(device)

    use_mixed_precision = optim_config.mixed_precision and device.type == "cuda"
    if use_mixed_precision and _uses_fft_spectral_modules(model):
        print(
            "warning: disabling CUDA mixed precision because the model uses complex FFT spectral modules. "
            "This avoids hard crashes/segmentation faults seen on some PyTorch+CUDA builds."
        )
        use_mixed_precision = False

    optimizer = AdamW(
        model.parameters(),
        lr=optim_config.learning_rate,
        weight_decay=optim_config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=optim_config.epochs,
        eta_min=optim_config.min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    history = []
    best_val = float("inf")
    checkpoint_dir = outputs["results"] / "checkpoints"

    for epoch in range(1, optim_config.epochs + 1):
        print(f"\nepoch={epoch:03d}/{optim_config.epochs:03d}")
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=optim_config.grad_clip_norm,
            log_every=optim_config.log_every,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            grad_clip_norm=None,
            log_every=optim_config.log_every,
        )
        scheduler.step()

        epoch_row = {
            "epoch": epoch,
            "train_loss_total": train_metrics["loss_total"],
            "train_loss_base": train_metrics["loss_base"],
            "train_loss_frequency": train_metrics["loss_frequency"],
            "train_rel_l1": train_metrics["rel_l1"],
            "train_mse": train_metrics["mse"],
            "val_loss_total": val_metrics["loss_total"],
            "val_loss_base": val_metrics["loss_base"],
            "val_loss_frequency": val_metrics["loss_frequency"],
            "val_rel_l1": val_metrics["rel_l1"],
            "val_mse": val_metrics["mse"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)
        print(
            "epoch_summary "
            f"train_rel_l1={train_metrics['rel_l1']:.6f} "
            f"val_rel_l1={val_metrics['rel_l1']:.6f} "
            f"val_mse={val_metrics['mse']:.6f}"
        )

        latest_path = checkpoint_dir / f"{optim_config.run_name}_latest.pt"
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            metrics=epoch_row,
            config=config,
        )

        if val_metrics["rel_l1"] < best_val:
            best_val = val_metrics["rel_l1"]
            best_path = checkpoint_dir / f"{optim_config.run_name}_best.pt"
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=epoch_row,
                config=config,
            )

        if epoch % optim_config.save_every == 0:
            epoch_path = checkpoint_dir / f"{optim_config.run_name}_epoch_{epoch:03d}.pt"
            save_checkpoint(
                epoch_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=epoch_row,
                config=config,
            )

    history_path = outputs["results"] / f"{optim_config.run_name}_history.csv"
    save_csv(history_path, history)
    plot_training_history(history, outputs["plots"] / f"{optim_config.run_name}_loss_curve.png")

    one_step_metrics = evaluate_one_step(model, val_loader, device=device)
    rollout_metrics = evaluate_rollout(model, rollout_loader, device=device)
    summary = {
        "run_name": optim_config.run_name,
        "best_val_rel_l1": best_val,
        "one_step": one_step_metrics,
        "rollout": rollout_metrics,
        "history_csv": str(history_path),
        "best_checkpoint": str(checkpoint_dir / f"{optim_config.run_name}_best.pt"),
    }
    save_json(outputs["results"] / f"{optim_config.run_name}_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DRIFT-NET on synthetic PDE surrogate data.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device string, e.g. auto, cpu, cuda.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional override for the run name.")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for the random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    if args.seed is not None:
        config["data"]["seed"] = args.seed
    if args.run_name is not None:
        config["training"]["run_name"] = args.run_name

    seed_everything(int(config["data"]["seed"]))
    device = resolve_device(args.device)
    summary = train_model(config=config, device=device)
    print("\ntraining_complete")
    print(summary)


if __name__ == "__main__":
    main()
