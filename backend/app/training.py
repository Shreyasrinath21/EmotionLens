from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .audio import AudioInputError, TARGET_SAMPLE_RATE, _model_features, decode_wav
from .experiments import calibration_report, evaluate_noise_robustness
from .inference import ARTIFACTS_DIR, CHECKPOINT_PATH
from .models import CLASS_LABELS, CREMA_D_CLASS_LABELS, MODEL_ARCHITECTURES, create_model


DATASET_LABELS = {
    "ravdess": ("RAVDESS", CLASS_LABELS),
    "crema-d": ("CREMA-D", CREMA_D_CLASS_LABELS),
}


def _read_ravdess(root: Path) -> list[tuple[Path, int, str]]:
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    records: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*.wav")):
        pieces = path.stem.split("-")
        if len(pieces) != 7 or pieces[0] != "03":
            continue
        try:
            class_index = int(pieces[2]) - 1
        except ValueError:
            continue
        filename_actor = pieces[-1].removeprefix("Actor_")
        directory_actor = path.parent.name.removeprefix("Actor_")
        actor_id = filename_actor if filename_actor.isdigit() else directory_actor
        if not actor_id.isdigit() or not 1 <= int(actor_id) <= 24:
            continue
        if 0 <= class_index < len(CLASS_LABELS):
            records.append((path, class_index, f"{int(actor_id):02d}"))
    if not records:
        raise ValueError(
            f"No labelled RAVDESS WAV files were found beneath {root}. "
            "Expected audio-only speech files such as "
            "Actor_01\\03-01-03-01-01-01-01.wav."
        )
    return records


def _read_crema_d(root: Path) -> list[tuple[Path, int, str]]:
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    emotion_indexes = {
        "NEU": CREMA_D_CLASS_LABELS.index("neutral"),
        "HAP": CREMA_D_CLASS_LABELS.index("happy"),
        "SAD": CREMA_D_CLASS_LABELS.index("sad"),
        "ANG": CREMA_D_CLASS_LABELS.index("angry"),
        "FEA": CREMA_D_CLASS_LABELS.index("fearful"),
        "DIS": CREMA_D_CLASS_LABELS.index("disgust"),
    }
    records: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*.wav")):
        pieces = path.stem.upper().split("_")
        if len(pieces) != 4 or not pieces[0].isdigit():
            continue
        actor_id, _, emotion, intensity = pieces
        if len(actor_id) != 4 or emotion not in emotion_indexes:
            continue
        if intensity not in {"LO", "MD", "HI", "XX"}:
            continue
        records.append((path, emotion_indexes[emotion], actor_id))
    if not records:
        raise ValueError(
            f"No labelled CREMA-D WAV files were found beneath {root}. "
            "Expected files such as 1001_IEO_ANG_HI.wav."
        )
    return records


def _read_dataset(
    root: Path, dataset_format: str
) -> tuple[list[tuple[Path, int, str]], str, tuple[str, ...]]:
    try:
        dataset_name, labels = DATASET_LABELS[dataset_format]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset format: {dataset_format}") from exc
    records = (
        _read_ravdess(root)
        if dataset_format == "ravdess"
        else _read_crema_d(root)
    )
    return records, dataset_name, labels


def _speaker_splits(
    records: list[tuple[Path, int, str]], seed: int
) -> dict[str, list[tuple[Path, int, str]]]:
    speakers = sorted({record[2] for record in records})
    if len(speakers) < 3:
        raise ValueError("At least three distinct speaker IDs are required for speaker splits.")
    generator = random.Random(seed)
    generator.shuffle(speakers)
    test_count = max(1, round(len(speakers) * 0.15))
    validation_count = max(1, round(len(speakers) * 0.15))
    if test_count + validation_count >= len(speakers):
        raise ValueError("Not enough distinct speakers for train/validation/test splits.")
    test_speakers = set(speakers[:test_count])
    validation_speakers = set(speakers[test_count : test_count + validation_count])
    train_speakers = set(speakers[test_count + validation_count :])
    splits = {
        "train": [item for item in records if item[2] in train_speakers],
        "validation": [item for item in records if item[2] in validation_speakers],
        "test": [item for item in records if item[2] in test_speakers],
    }
    memberships = {name: {item[2] for item in entries} for name, entries in splits.items()}
    if (
        memberships["train"] & memberships["validation"]
        or memberships["train"] & memberships["test"]
        or memberships["validation"] & memberships["test"]
    ):
        raise RuntimeError("Speaker overlap detected in the generated splits.")
    return splits


def _class_counts(
    records: list[tuple[Path, int, str]],
    class_labels: tuple[str, ...] = CLASS_LABELS,
) -> dict[str, int]:
    return {
        label: sum(record[1] == index for record in records)
        for index, label in enumerate(class_labels)
    }


def audit_dataset(root: Path, dataset_format: str) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    records, dataset_name, class_labels = _read_dataset(root, dataset_format)
    all_wavs = sorted(root.rglob("*.wav"))
    record_by_path = {path.resolve(): (label, speaker) for path, label, speaker in records}
    class_counts: Counter[str] = Counter()
    speakers: set[str] = set()
    sample_rates: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    durations: list[float] = []
    invalid: list[dict[str, str]] = []
    invalid_paths: set[Path] = set()
    valid_count = 0
    for path in all_wavs:
        labelled_record = record_by_path.get(path.resolve())
        if labelled_record is None:
            continue
        try:
            audio = decode_wav(path.read_bytes())
        except (OSError, AudioInputError) as exc:
            invalid_paths.add(path.resolve())
            if len(invalid) < 25:
                invalid.append({"file": str(path.relative_to(root)), "error": str(exc)})
            continue
        label_index, speaker = labelled_record
        valid_count += 1
        class_counts[class_labels[label_index]] += 1
        speakers.add(speaker)
        sample_rates[str(audio.original_sample_rate)] += 1
        channels[str(audio.original_channels)] += 1
        durations.append(audio.original_duration)
    if valid_count == 0:
        raise ValueError(
            f"No valid labelled {dataset_name} speech WAV files were found beneath {root}."
        )
    split_audit: dict[str, list[str]] | None = None
    if len(speakers) >= 3:
        valid_records = [
            record for record in records if record[0].resolve() not in invalid_paths
        ]
        splits = _speaker_splits(valid_records, seed=42)
        split_audit = {
            name: sorted({entry[2] for entry in entries})
            for name, entries in splits.items()
        }
    return {
        "status": "audited",
        "dataset": dataset_name,
        "dataset_root": str(root.resolve()),
        "total_wav_files": len(all_wavs),
        "recognized_labelled_files": len(records),
        "valid_labelled_files": valid_count,
        "invalid_labelled_file_count": len(records) - valid_count,
        "invalid_labelled_file_examples": invalid,
        "unrecognized_or_unlabelled_wav_count": len(all_wavs) - len(records),
        "speaker_count": len(speakers),
        "class_counts": {label: class_counts[label] for label in class_labels},
        "sample_rate_counts_hz": dict(sample_rates),
        "channel_counts": dict(channels),
        "duration_seconds": {
            "minimum": min(durations),
            "median": float(np.median(durations)),
            "maximum": max(durations),
        },
        "speaker_split_audit_seed_42": split_audit,
        "limitations": [
            "Filename parsers only recognize the configured dataset and speech-label format.",
            "Energy validation rejects malformed, silent, unsupported-rate, and over-length WAVs; it does not assess label correctness or licensing.",
        ],
    }


def evaluate_cross_corpus(
    checkpoint_dir: Path,
    dataset_dir: Path,
    dataset_format: str,
) -> dict[str, Any]:
    metadata = json.loads((checkpoint_dir / "model.json").read_text(encoding="utf-8"))
    source_dataset = str(metadata["dataset"])
    records, target_dataset, target_labels = _read_dataset(dataset_dir, dataset_format)
    if source_dataset.casefold() == target_dataset.casefold():
        raise ValueError("Cross-corpus evaluation requires a different target dataset.")
    source_labels = tuple(metadata["class_labels"])
    shared_labels = tuple(label for label in CLASS_LABELS if label in source_labels and label in target_labels)
    if len(shared_labels) < 2:
        raise ValueError("Source and target datasets must share at least two emotion labels.")
    source_indexes = {label: index for index, label in enumerate(source_labels)}
    selected = [
        record for record in records if target_labels[record[1]] in shared_labels
    ]
    if not selected:
        raise ValueError("No target recordings use the source model's shared emotion labels.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(metadata["architecture"], len(source_labels)).to(device)
    state = torch.load(
        checkpoint_dir / "model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    probabilities: list[np.ndarray] = []
    source_confidences: list[float] = []
    truth: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for path, target_index, _ in selected:
            decoded = decode_wav(path.read_bytes())
            feature = _model_features(decoded.samples)
            scores = torch.softmax(
                model(
                    torch.from_numpy(feature)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                ),
                dim=1,
            )[0].cpu().numpy()
            shared_scores = np.asarray(
                [scores[source_indexes[label]] for label in shared_labels]
            )
            shared_total = float(shared_scores.sum())
            if shared_total <= 0 or not np.isfinite(shared_total):
                raise RuntimeError("The source model returned invalid cross-corpus scores.")
            probabilities.append(shared_scores / shared_total)
            source_confidences.append(float(scores.max()))
            truth.append(
                shared_labels.index(target_labels[target_index])
            )
    latency_ms = (time.perf_counter() - started) * 1000 / len(selected)
    probability_array = np.asarray(probabilities)
    truth_array = np.asarray(truth)
    predicted = probability_array.argmax(axis=1)
    threshold = float(metadata["uncertainty_threshold"])
    confidence = np.asarray(source_confidences)
    accepted = confidence >= threshold
    return {
        "status": "evaluated",
        "source_dataset": source_dataset,
        "source_model_version": metadata["model_version"],
        "target_dataset": target_dataset,
        "target_dataset_format": dataset_format,
        "evaluation_protocol": "Train on source corpus; evaluate once on the full separate target corpus. No target fine-tuning or threshold selection.",
        "shared_class_labels": list(shared_labels),
        "excluded_target_class_labels": [
            label for label in target_labels if label not in shared_labels
        ],
        "target_speaker_count": len({record[2] for record in selected}),
        "target_sample_count": len(selected),
        "accuracy": float(accuracy_score(truth_array, predicted)),
        "macro_precision": float(
            precision_score(truth_array, predicted, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth_array, predicted, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(truth_array, predicted, average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            truth_array,
            predicted,
            labels=list(range(len(shared_labels))),
            target_names=list(shared_labels),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            truth_array, predicted, labels=list(range(len(shared_labels)))
        ).tolist(),
        "target_class_counts": {
            label: int(np.count_nonzero(truth_array == index))
            for index, label in enumerate(shared_labels)
        },
        "source_validation_threshold": threshold,
        "selective_coverage": float(np.mean(accepted)),
        "selective_accuracy": float(np.mean(predicted[accepted] == truth_array[accepted]))
        if accepted.any()
        else None,
        "target_expected_calibration_error_on_shared_labels": calibration_report(
            probability_array, truth_array
        )["expected_calibration_error"],
        "preprocessing_and_inference_latency_ms_per_clip": latency_ms,
        "note": "Cross-corpus scores measure dataset transfer on an external corpus; the differing recording and label conventions may affect comparability.",
    }


class AudioEmotionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[tuple[Path, int, str]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label, _ = self.records[index]
        try:
            contents = path.read_bytes()
            decoded = decode_wav(contents)
        except (OSError, AudioInputError) as exc:
            raise RuntimeError(f"Could not load dataset audio {path.name}: {exc}") from exc
        feature = _model_features(decoded.samples)
        return torch.from_numpy(feature).unsqueeze(0), torch.tensor(label, dtype=torch.long)


def _predict(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[list[int], np.ndarray]:
    model.eval()
    predicted: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for features, _ in loader:
            logits = model(features.to(device))
            scores = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.extend(scores)
            predicted.extend(scores.argmax(axis=1).tolist())
    return predicted, np.asarray(probabilities)


def _choose_threshold(probabilities: np.ndarray, truth: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    best_threshold = 1.01
    best_coverage = 0.0
    for threshold in np.linspace(0.0, 1.0, 101):
        selected = confidence >= threshold
        if not selected.any():
            continue
        selective_accuracy = float(np.mean(predictions[selected] == truth[selected]))
        coverage = float(np.mean(selected))
        if selective_accuracy >= 0.75 and coverage > best_coverage:
            best_threshold = float(threshold)
            best_coverage = coverage
    return best_threshold


def train(
    dataset_dir: Path,
    architecture: str,
    epochs: int,
    batch_size: int,
    seed: int,
    robustness: bool = False,
    dataset_version: str = "unspecified",
    dataset_source: str = "Local dataset copy; source URL not recorded",
    output_dir: Path | None = None,
    dataset_format: str = "ravdess",
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, dataset_name, class_labels = _read_dataset(dataset_dir, dataset_format)
    splits = _speaker_splits(records, seed)
    loaders = {
        name: DataLoader(
            AudioEmotionDataset(entries),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, entries in splits.items()
    }
    model = create_model(architecture, len(class_labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for features, labels in loaders["train"]:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_examples += len(labels)
        model.eval()
        validation_loss = 0.0
        validation_examples = 0
        with torch.inference_mode():
            for features, labels in loaders["validation"]:
                features, labels = features.to(device), labels.to(device)
                loss = criterion(model(features), labels)
                validation_loss += float(loss.item()) * len(labels)
                validation_examples += len(labels)
        train_loss = total_loss / max(total_examples, 1)
        validation_loss /= max(validation_examples, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    _, validation_probabilities = _predict(
        model, DataLoader(AudioEmotionDataset(splits["validation"]), batch_size=batch_size), device
    )
    validation_truth = np.asarray([entry[1] for entry in splits["validation"]])
    threshold = _choose_threshold(validation_probabilities, validation_truth)

    test_loader = DataLoader(AudioEmotionDataset(splits["test"]), batch_size=batch_size)
    started = time.perf_counter()
    test_predictions, test_probabilities = _predict(model, test_loader, device)
    latency_ms_per_example = (
        (time.perf_counter() - started) * 1000 / len(splits["test"])
    )
    test_truth = np.asarray([entry[1] for entry in splits["test"]])
    test_confidence = test_probabilities.max(axis=1)
    selected = test_confidence >= threshold

    output_dir = output_dir or ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = dataset_name.lower().replace("-", "")
    version = datetime.now(timezone.utc).strftime(
        f"{architecture}-{dataset_slug}-%Y%m%dT%H%M%SZ"
    )
    torch.save(best_state, output_dir / "model.pt")
    metadata = {
        "model_version": version,
        "checkpoint_file": CHECKPOINT_PATH.name,
        "architecture": architecture,
        "class_labels": list(class_labels),
        "uncertainty_threshold": threshold,
        "threshold_selection": "Maximum validation coverage with at least 75% selective accuracy; validation only.",
        "dataset": dataset_name,
        "dataset_version": dataset_version,
        "dataset_source": dataset_source,
        "dataset_file_count": len(records),
        "dataset_audit": {
            "speaker_count": len({record[2] for record in records}),
            "class_counts": _class_counts(records, class_labels),
            "split_class_counts": {
                name: _class_counts(entries, class_labels)
                for name, entries in splits.items()
            },
        },
        "experiment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "split_speakers": {
            name: sorted({item[2] for item in entries}) for name, entries in splits.items()
        },
        "split_sizes": {name: len(entries) for name, entries in splits.items()},
        "preprocessing": {
            "sample_rate_hz": TARGET_SAMPLE_RATE,
            "mono": True,
            "peak_normalization": True,
            "feature": "64-band log-Mel spectrogram",
            "window_seconds": 3,
            "hop_seconds": 2,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "early_stopping_patience": 5,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "best_validation_loss": best_validation_loss,
            "history": history,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "librosa_version": librosa.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "limitations": [
            (
                "RAVDESS consists of acted English speech and is not representative of all speakers or contexts."
                if dataset_name == "RAVDESS"
                else "CREMA-D consists of acted English speech and is not representative of all speakers or contexts."
            ),
            "Softmax scores are not calibrated.",
            "Predictions are not evidence of a person's internal emotional state.",
        ],
    }
    (output_dir / "model.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    # Store a speaker audit without recording speaker names or audio.
    speaker_audit = {
        name: sorted({item[2] for item in entries}) for name, entries in splits.items()
    }
    metric_predictions = np.asarray(test_predictions)
    report = {
        "status": "evaluated",
        "model_version": version,
        "dataset": dataset_name,
        "split_method": "speaker-independent",
        "speaker_overlap": False,
        "speaker_split_audit": speaker_audit,
        "test_class_counts": _class_counts(splits["test"], class_labels),
        "test_sample_count": len(test_truth),
        "accuracy": float(accuracy_score(test_truth, metric_predictions)),
        "macro_precision": float(
            precision_score(test_truth, metric_predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(test_truth, metric_predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(test_truth, metric_predictions, average="macro", zero_division=0)),
        "per_class": classification_report(
            test_truth,
            metric_predictions,
            labels=list(range(len(class_labels))),
            target_names=list(class_labels),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            test_truth, metric_predictions, labels=list(range(len(class_labels)))
        ).tolist(),
        "class_labels": list(class_labels),
        "uncertainty_threshold": threshold,
        "selective_coverage": float(np.mean(selected)),
        "selective_accuracy": float(np.mean(metric_predictions[selected] == test_truth[selected]))
        if selected.any()
        else None,
        "test_preprocessing_and_inference_latency_ms_per_clip": latency_ms_per_example,
        "calibration": calibration_report(test_probabilities, test_truth),
        "note": "Metrics are computed only when this training command completes; test speakers were held out.",
    }
    if robustness:
        report["robustness"] = evaluate_noise_robustness(
            model,
            splits["test"],
            device=device,
            seed=seed,
            class_labels=class_labels,
        )
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if robustness:
        (output_dir / "robustness.json").write_text(
            json.dumps(report["robustness"], indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate an EmotionLens model.")
    parser.add_argument("--dataset", type=Path, required=True, help="Local dataset directory.")
    parser.add_argument(
        "--dataset-format",
        choices=tuple(DATASET_LABELS),
        default="ravdess",
        help="Dataset filename format and emotion mapping.",
    )
    parser.add_argument("--architecture", choices=MODEL_ARCHITECTURES, default="cnn_lstm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", default="unspecified")
    parser.add_argument(
        "--dataset-source",
        default="Local dataset copy; source URL not recorded",
    )
    parser.add_argument(
        "--audit-dataset",
        action="store_true",
        help="Validate a local corpus and print label, speaker, duration, channel, and sample-rate audit JSON.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ARTIFACTS_DIR / "dataset_audit.json",
        help="Path to save the dataset audit JSON (default: backend/artifacts/dataset_audit.json).",
    )
    parser.add_argument(
        "--cross-corpus-dataset",
        type=Path,
        help="Optional external corpus directory for a no-finetuning transfer evaluation.",
    )
    parser.add_argument(
        "--cross-corpus-format",
        choices=tuple(DATASET_LABELS),
        help="Filename format for --cross-corpus-dataset.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train CNN, CNN-LSTM, and CNN-BiLSTM-attention on identical speaker splits.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run clean/noise/denoising experiments on the held-out test split.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers.")
    if args.audit_dataset:
        try:
            audit = audit_dataset(args.dataset, args.dataset_format)
        except ValueError as exc:
            parser.error(str(exc))
        serialized = json.dumps(audit, indent=2)
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(serialized, encoding="utf-8")
        print(serialized)
        return
    if (args.cross_corpus_dataset is None) != (args.cross_corpus_format is None):
        parser.error("--cross-corpus-dataset and --cross-corpus-format must be used together.")
    if args.compare:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        architectures = (
            args.architecture,
            *(item for item in MODEL_ARCHITECTURES if item != args.architecture),
        )
        reports = {}
        for architecture in architectures:
            reports[architecture] = train(
                args.dataset,
                architecture,
                args.epochs,
                args.batch_size,
                args.seed,
                robustness=args.robustness,
                dataset_version=args.dataset_version,
                dataset_source=args.dataset_source,
                output_dir=ARTIFACTS_DIR / "models" / architecture,
                dataset_format=args.dataset_format,
            )
        primary_dir = ARTIFACTS_DIR / "models" / args.architecture
        for filename in ("model.pt", "model.json", "evaluation.json"):
            shutil.copy2(primary_dir / filename, ARTIFACTS_DIR / filename)
        if args.robustness:
            shutil.copy2(primary_dir / "robustness.json", ARTIFACTS_DIR / "robustness.json")
        else:
            (ARTIFACTS_DIR / "robustness.json").unlink(missing_ok=True)
        comparison = {
            "status": "evaluated",
            "dataset": DATASET_LABELS[args.dataset_format][0],
            "split_method": "identical speaker-independent split",
            "random_seed": args.seed,
            "selected_primary_model": args.architecture,
            "models": {
                architecture: {
                    key: reports[architecture][key]
                    for key in (
                        "model_version",
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "test_sample_count",
                        "test_preprocessing_and_inference_latency_ms_per_clip",
                    )
                }
                for architecture in architectures
            },
            "note": "Each score was measured in a separate training run using the same speaker IDs and seed.",
        }
        (ARTIFACTS_DIR / "comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
        report = reports[args.architecture]
    else:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        report = train(
            args.dataset,
            args.architecture,
            args.epochs,
            args.batch_size,
            args.seed,
            robustness=args.robustness,
            dataset_version=args.dataset_version,
            dataset_source=args.dataset_source,
            dataset_format=args.dataset_format,
        )
        (ARTIFACTS_DIR / "comparison.json").unlink(missing_ok=True)
        (ARTIFACTS_DIR / "robustness.json").unlink(missing_ok=True)
    if args.cross_corpus_dataset is not None and args.cross_corpus_format is not None:
        source_checkpoint_dir = (
            ARTIFACTS_DIR / "models" / args.architecture
            if args.compare
            else ARTIFACTS_DIR
        )
        try:
            cross_report = evaluate_cross_corpus(
                source_checkpoint_dir,
                args.cross_corpus_dataset,
                args.cross_corpus_format,
            )
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            parser.error(f"Cross-corpus evaluation failed: {exc}")
        cross_report["source_dataset_version"] = args.dataset_version
        (ARTIFACTS_DIR / "cross_corpus.json").write_text(
            json.dumps(cross_report, indent=2), encoding="utf-8"
        )
        print(json.dumps({"training": report, "cross_corpus": cross_report}, indent=2))
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
