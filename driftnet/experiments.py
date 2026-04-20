from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .datasets import PDEDataConfig
from .losses import DriftLoss
from .model import (
    DRIFTNet,
    DriftNetConfig,
    ScOTBaseline,
    ScOTConfig,
)
from .train import train_model
from .utils import (
    benchmark_train_step,
    count_parameters,
    ensure_output_dirs,
    load_yaml_config,
    peak_memory_megabytes,
    plot_series,
    save_csv,
    save_json,
    seed_everything,
    throughput_benchmark,
)


def clone_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(base_config)


def drift_model_from_config(config: Dict[str, Any]) -> DRIFTNet:
    return DRIFTNet(DriftNetConfig.from_dict(config["model"]))

def scot_model_from_config(config: Dict[str, Any]) -> ScOTBaseline:
    scot_section = dict(config.get("scot", {}))
    scot_section.setdefault("in_channels", config["model"]["in_channels"])
    scot_section.setdefault("out_channels", config["model"]["out_channels"])
    return ScOTBaseline(ScOTConfig(**scot_section))


def run_ablation_study(base_config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    outputs = ensure_output_dirs()
    experiments: List[Tuple[str, Dict[str, Any]]] = []
    ablation_epochs = int(base_config["experiments"]["ablation"]["epochs"])

    full = clone_config(base_config)
    full["training"]["run_name"] = "ablation_full_driftnet"
    full["training"]["epochs"] = ablation_epochs
    experiments.append(("Full DRIFTNet", full))

    no_low_mix = clone_config(base_config)
    no_low_mix["training"]["run_name"] = "ablation_no_low_mix"
    no_low_mix["training"]["epochs"] = ablation_epochs
    no_low_mix["model"]["enable_low_frequency_mixing"] = False
    experiments.append(("Without Low Frequency Mixing", no_low_mix))

    no_gate = clone_config(base_config)
    no_gate["training"]["run_name"] = "ablation_no_radial_gate"
    no_gate["training"]["epochs"] = ablation_epochs
    no_gate["model"]["enable_radial_gating"] = False
    experiments.append(("Without Radial Gating", no_gate))

    no_freq_loss = clone_config(base_config)
    no_freq_loss["training"]["run_name"] = "ablation_no_freq_loss"
    no_freq_loss["training"]["epochs"] = ablation_epochs
    no_freq_loss["loss"]["lambda_frequency"] = 0.0
    experiments.append(("Without Frequency Weighted Loss", no_freq_loss))

    local_only = clone_config(base_config)
    local_only["training"]["run_name"] = "ablation_local_only"
    local_only["training"]["epochs"] = ablation_epochs
    local_only["model"]["enable_spectral_branch"] = False
    experiments.append(("Local Branch Only", local_only))

    spectral_only = clone_config(base_config)
    spectral_only["training"]["run_name"] = "ablation_spectral_only"
    spectral_only["training"]["epochs"] = ablation_epochs
    spectral_only["model"]["enable_local_branch"] = False
    experiments.append(("Spectral Branch Only", spectral_only))

    rows = []
    summaries = {}
    for label, config in experiments:
        print(f"\nrunning_ablation={label}")
        summary = train_model(config=config, device=device)
        summaries[label] = summary
        rows.append(
            {
                "variant": label,
                "val_relative_l1": summary["one_step"]["relative_l1"],
                "val_mse": summary["one_step"]["mse"],
                "rollout_relative_l1_final": summary["rollout"]["relative_l1_final"],
            }
        )

    table_path = outputs["tables"] / "ablation_relative_l1.csv"
    save_csv(table_path, rows)
    save_json(outputs["results"] / "ablation_summary.json", summaries)
    return {"table_path": str(table_path), "rows": rows, "summaries": summaries}


def run_rollout_drift_test(base_config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    outputs = ensure_output_dirs()
    rollout_horizon = int(base_config["experiments"]["rollout"]["horizon"])

    drift_cfg = clone_config(base_config)
    drift_cfg["training"]["run_name"] = "rollout_driftnet"
    drift_cfg["data"]["rollout_horizon"] = rollout_horizon

    scot_cfg = clone_config(base_config)
    scot_cfg["training"]["run_name"] = "rollout_scot_baseline"
    scot_cfg["data"]["rollout_horizon"] = rollout_horizon

    drift_summary = train_model(config=drift_cfg, device=device, model=drift_model_from_config(drift_cfg))
    scot_summary = train_model(config=scot_cfg, device=device, model=scot_model_from_config(scot_cfg))

    x_axis = list(range(1, len(drift_summary["rollout"]["relative_l1_per_step"]) + 1))
    plot_path = outputs["plots"] / "rollout_drift_comparison.png"
    plot_series(
        x_values=x_axis,
        y_series={
            "DRIFTNet": drift_summary["rollout"]["relative_l1_per_step"],
            "scOT": scot_summary["rollout"]["relative_l1_per_step"],
        },
        title="Long-Horizon Rollout Drift",
        x_label="Autoregressive Step",
        y_label="Relative L1",
        path=plot_path,
    )

    rows = []
    for name, summary in [("DRIFTNet", drift_summary), ("scOT", scot_summary)]:
        rows.append(
            {
                "model": name,
                "rollout_relative_l1_mean": summary["rollout"]["relative_l1_mean"],
                "rollout_relative_l1_final": summary["rollout"]["relative_l1_final"],
                "one_step_relative_l1": summary["one_step"]["relative_l1"],
            }
        )

    table_path = outputs["tables"] / "rollout_drift_metrics.csv"
    save_csv(table_path, rows)
    return {
        "plot_path": str(plot_path),
        "table_path": str(table_path),
        "rows": rows,
    }


def run_efficiency_benchmark(base_config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    outputs = ensure_output_dirs()
    model_config = DriftNetConfig.from_dict(base_config["model"])
    data_config = PDEDataConfig.from_dict(base_config["data"])
    loss = DriftLoss(
        lambda_frequency=base_config["loss"]["lambda_frequency"],
        frequency_power=base_config["loss"]["frequency_power"],
        frequency_bias=base_config["loss"]["frequency_bias"],
    )

    drift = DRIFTNet(model_config).to(device)
    scot = scot_model_from_config(base_config).to(device)

    input_shape = (
        int(base_config["training"]["batch_size"]),
        model_config.in_channels,
        data_config.image_size,
        data_config.image_size,
    )

    rows = []
    for name, model in [("DRIFTNet", drift), ("scOT", scot)]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_timing = benchmark_train_step(
            model=model,
            criterion=loss,
            input_shape=input_shape,
            device=device,
            warmup_steps=int(base_config["experiments"]["efficiency"]["warmup_steps"]),
            benchmark_steps=int(base_config["experiments"]["efficiency"]["benchmark_steps"]),
        )
        inference_timing = throughput_benchmark(
            model=model,
            input_shape=input_shape,
            device=device,
            warmup_steps=int(base_config["experiments"]["efficiency"]["warmup_steps"]),
            benchmark_steps=int(base_config["experiments"]["efficiency"]["benchmark_steps"]),
        )
        rows.append(
            {
                "model": name,
                "parameters": count_parameters(model),
                "train_step_seconds": train_timing["train_step_seconds"],
                "samples_per_second": inference_timing["samples_per_second"],
                "peak_gpu_memory_mb": peak_memory_megabytes(device),
            }
        )

    table_path = outputs["tables"] / "efficiency_benchmark.csv"
    save_csv(table_path, rows)
    return {"table_path": str(table_path), "rows": rows}


def run_surrogate_result(base_config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    outputs = ensure_output_dirs()
    drift_cfg = clone_config(base_config)
    drift_cfg["training"]["run_name"] = "surrogate_driftnet"

    scot_cfg = clone_config(base_config)
    scot_cfg["training"]["run_name"] = "surrogate_scot_baseline"

    drift_summary = train_model(config=drift_cfg, device=device, model=drift_model_from_config(drift_cfg))
    scot_summary = train_model(config=scot_cfg, device=device, model=scot_model_from_config(scot_cfg))

    rows = [
        {
            "model": "DRIFTNet",
            "one_step_relative_l1": drift_summary["one_step"]["relative_l1"],
            "rollout_relative_l1_final": drift_summary["rollout"]["relative_l1_final"],
        },
        {
            "model": "scOT",
            "one_step_relative_l1": scot_summary["one_step"]["relative_l1"],
            "rollout_relative_l1_final": scot_summary["rollout"]["relative_l1_final"],
        },
    ]

    table_path = outputs["tables"] / "surrogate_forecasting_comparison.csv"
    save_csv(table_path, rows)
    save_json(
        outputs["results"] / "surrogate_forecasting_comparison.json",
        {
            "DRIFTNet": drift_summary,
            "scOT": scot_summary,
        },
    )
    return {"table_path": str(table_path), "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DRIFT-NET paper-style experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "ablation", "rollout", "efficiency", "surrogate"],
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    seed_everything(int(config["data"]["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    results: Dict[str, Any] = {}
    if args.experiment in {"all", "ablation"}:
        results["ablation"] = run_ablation_study(config, device=device)
    if args.experiment in {"all", "rollout"}:
        results["rollout"] = run_rollout_drift_test(config, device=device)
    if args.experiment in {"all", "efficiency"}:
        results["efficiency"] = run_efficiency_benchmark(config, device=device)
    if args.experiment in {"all", "surrogate"}:
        results["surrogate"] = run_surrogate_result(config, device=device)

    save_json(ensure_output_dirs()["results"] / "experiment_manifest.json", results)
    print(results)


if __name__ == "__main__":
    main()
