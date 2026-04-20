from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .datasets import PDEDataConfig, SyntheticPDEPairsDataset, build_datasets
from .utils import ensure_output_dirs, load_yaml_config, save_json, seed_everything


def _serialize_field(values) -> str:
    flat = values.reshape(-1).tolist()
    return " ".join(f"{value:.8e}" for value in flat)


def export_pairs_dataset_to_csv(
    dataset: SyntheticPDEPairsDataset,
    csv_path: str | Path,
    split_name: str,
    image_size: int,
) -> Dict[str, int]:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "sample_index",
                "channels",
                "height",
                "width",
                "input_field",
                "target_field",
            ]
        )
        for sample_index, (input_field, target_field) in enumerate(dataset):
            writer.writerow(
                [
                    split_name,
                    sample_index,
                    int(input_field.shape[0]),
                    image_size,
                    image_size,
                    _serialize_field(input_field),
                    _serialize_field(target_field),
                ]
            )

    return {"num_samples": len(dataset)}


def export_synthetic_navier_stokes_csv(config: Dict, output_dir: str | Path) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = PDEDataConfig.from_dict(config["data"])
    navier_stokes_config = PDEDataConfig(**{**data_config.__dict__, "dataset_name": "navier_stokes"})

    seed_everything(int(navier_stokes_config.seed))
    train_dataset, val_dataset, rollout_dataset = build_datasets(navier_stokes_config)

    train_csv = output_dir / "navier_stokes_train.csv"
    val_csv = output_dir / "navier_stokes_val.csv"

    train_summary = export_pairs_dataset_to_csv(
        dataset=train_dataset,
        csv_path=train_csv,
        split_name="train",
        image_size=navier_stokes_config.image_size,
    )
    val_summary = export_pairs_dataset_to_csv(
        dataset=val_dataset,
        csv_path=val_csv,
        split_name="val",
        image_size=navier_stokes_config.image_size,
    )

    metadata = {
        "dataset_name": "navier_stokes",
        "image_size": navier_stokes_config.image_size,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "train_samples": train_summary["num_samples"],
        "val_samples": val_summary["num_samples"],
        "rollout_trajectories": len(rollout_dataset),
        "format": {
            "input_field": "single string column containing flattened CxHxW values separated by spaces",
            "target_field": "single string column containing flattened CxHxW values separated by spaces",
        },
    }
    save_json(output_dir / "navier_stokes_metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export synthetic Navier-Stokes surrogate data to CSV.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Directory where the CSV files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    metadata = export_synthetic_navier_stokes_csv(config=config, output_dir=args.output_dir)
    print(metadata)


if __name__ == "__main__":
    main()
