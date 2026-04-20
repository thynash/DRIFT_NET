from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .datasets import PDEDataConfig, build_datasets
from .model import DRIFTNet, DriftNetConfig
from .utils import (
    count_parameters,
    ensure_output_dirs,
    load_checkpoint,
    load_yaml_config,
    mse,
    plot_series,
    relative_l1,
    rollout_autoregressive,
    save_csv,
    save_json,
    throughput_benchmark,
)


@torch.no_grad()
def evaluate_one_step(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    rel_l1_values: List[Tensor] = []
    mse_values: List[Tensor] = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(inputs)
        rel_l1_values.append(relative_l1(predictions, targets).cpu())
        mse_values.append(mse(predictions, targets).cpu())

    model.train(was_training)
    rel_l1_tensor = torch.cat(rel_l1_values, dim=0)
    mse_tensor = torch.cat(mse_values, dim=0)
    return {
        "relative_l1": float(rel_l1_tensor.mean().item()),
        "mse": float(mse_tensor.mean().item()),
    }


@torch.no_grad()
def evaluate_rollout(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    rel_l1_steps: List[Tensor] = []
    mse_steps: List[Tensor] = []

    for trajectory in loader:
        # trajectory: (B, T+1, C, H, W) because batch_size=1 during evaluation
        initial_state = trajectory[:, 0].to(device)       # (B, C, H, W)
        target_rollout = trajectory[:, 1:]                # (B, T, C, H, W)
        predictions = rollout_autoregressive(
            model=model,
            initial_state=initial_state,
            horizon=target_rollout.shape[1],
            device=device,
        )                                                 # (B, T, C, H, W)

        step_rel_l1 = []
        step_mse = []
        for step in range(target_rollout.shape[1]):
            step_rel_l1.append(relative_l1(predictions[:, step], target_rollout[:, step]).mean())
            step_mse.append(mse(predictions[:, step], target_rollout[:, step]).mean())
        rel_l1_steps.append(torch.stack(step_rel_l1))
        mse_steps.append(torch.stack(step_mse))

    rel_l1_matrix = torch.stack(rel_l1_steps, dim=0)  # (N_rollouts, T)
    mse_matrix = torch.stack(mse_steps, dim=0)        # (N_rollouts, T)
    return {
        "relative_l1_mean": float(rel_l1_matrix.mean().item()),
        "relative_l1_final": float(rel_l1_matrix[:, -1].mean().item()),
        "mse_mean": float(mse_matrix.mean().item()),
        "mse_final": float(mse_matrix[:, -1].mean().item()),
        "relative_l1_per_step": rel_l1_matrix.mean(dim=0).tolist(),
        "mse_per_step": mse_matrix.mean(dim=0).tolist(),
    }


def evaluate_checkpoint(config: Dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> Dict[str, Any]:
    outputs = ensure_output_dirs()

    model_config = DriftNetConfig.from_dict(config["model"])
    data_config = PDEDataConfig.from_dict(config["data"])
    model = DRIFTNet(model_config).to(device)
    load_checkpoint(checkpoint_path, model=model, map_location=device)

    _, val_dataset, rollout_dataset = build_datasets(data_config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    rollout_loader = DataLoader(rollout_dataset, batch_size=1, shuffle=False, num_workers=0)

    one_step = evaluate_one_step(model, val_loader, device=device)
    rollout = evaluate_rollout(model, rollout_loader, device=device)
    throughput = throughput_benchmark(
        model=model,
        input_shape=(config["training"]["batch_size"], model_config.in_channels, data_config.image_size, data_config.image_size),
        device=device,
        warmup_steps=int(config["experiments"]["efficiency"]["warmup_steps"]),
        benchmark_steps=int(config["experiments"]["efficiency"]["benchmark_steps"]),
    )

    metrics = {
        "checkpoint": str(checkpoint_path),
        "parameters": count_parameters(model),
        "one_step_relative_l1": one_step["relative_l1"],
        "one_step_mse": one_step["mse"],
        "rollout_relative_l1_mean": rollout["relative_l1_mean"],
        "rollout_relative_l1_final": rollout["relative_l1_final"],
        "rollout_mse_mean": rollout["mse_mean"],
        "rollout_mse_final": rollout["mse_final"],
        "samples_per_second": throughput["samples_per_second"],
    }

    per_step_rows = [
        {
            "step": step + 1,
            "relative_l1": value,
            "mse": rollout["mse_per_step"][step],
        }
        for step, value in enumerate(rollout["relative_l1_per_step"])
    ]
    save_csv(outputs["results"] / "evaluation_rollout_metrics.csv", per_step_rows)
    save_json(outputs["results"] / "evaluation_summary.json", metrics)
    plot_series(
        x_values=[row["step"] for row in per_step_rows],
        y_series={"relative_l1": [row["relative_l1"] for row in per_step_rows]},
        title="Autoregressive Rollout Error",
        x_label="Rollout Step",
        y_label="Relative L1",
        path=outputs["plots"] / "evaluation_rollout_relative_l1.png",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DRIFT-NET checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    metrics = evaluate_checkpoint(config=config, checkpoint_path=args.checkpoint, device=device)
    print(metrics)


if __name__ == "__main__":
    main()
