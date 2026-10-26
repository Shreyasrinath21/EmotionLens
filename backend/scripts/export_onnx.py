"""Export checked-in PyTorch checkpoints to the compact Vercel ONNX runtime format."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

from app.models import create_model


def export_model(model_dir: Path, checkpoint: Path | None = None) -> None:
    metadata_path = model_dir / "model.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    architecture = metadata["architecture"]
    labels = metadata["class_labels"]
    model = create_model(architecture, len(labels))
    weights = torch.load(
        checkpoint or model_dir / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(weights)
    model.eval()

    output_path = model_dir / "model.onnx"
    sample = torch.zeros((1, 1, 64, 300), dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["log_mel"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported {architecture} to {output_path}")


def main() -> None:
    export_model(ARTIFACTS, ARTIFACTS / "model.pt")
    for architecture in ("cnn", "cnn_lstm", "cnn_bilstm_attention"):
        export_model(ARTIFACTS / "models" / architecture)


if __name__ == "__main__":
    main()
