from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import ARTIFACTS_DIR
from .training import DATASET_LABELS, evaluate_cross_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing EmotionLens checkpoint on a separate corpus."
    )
    parser.add_argument("--dataset", required=True, help="Local target corpus directory.")
    parser.add_argument(
        "--dataset-format",
        choices=tuple(DATASET_LABELS),
        required=True,
        help="Filename format and label map for the target corpus.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ARTIFACTS_DIR),
        help="Source model directory containing model.json and model.pt.",
    )
    parser.add_argument(
        "--output",
        default=str(ARTIFACTS_DIR / "cross_corpus.json"),
        help="Path to save the measured cross-corpus report.",
    )
    args = parser.parse_args()
    try:
        report = evaluate_cross_corpus(
            Path(args.checkpoint_dir),
            Path(args.dataset),
            args.dataset_format,
        )
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.error(f"Cross-corpus evaluation failed: {exc}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
