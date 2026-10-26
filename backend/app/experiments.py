from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import functional as F

from .audio import AudioInputError, _model_features, decode_wav, denoise_wiener
from .models import CLASS_LABELS


def add_white_noise(samples: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    signal_rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if signal_rms == 0:
        raise AudioInputError("Noise robustness testing requires non-silent audio.")
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = generator.normal(0.0, noise_rms, size=samples.shape)
    noisy = samples.astype(np.float64) + noise
    peak = float(np.max(np.abs(noisy)))
    if peak > 0.99:
        noisy *= 0.99 / peak
    return noisy.astype(np.float32)


def evaluate_noise_robustness(
    model: torch.nn.Module,
    records: list[tuple[Path, int, str]],
    device: torch.device,
    seed: int = 42,
    class_labels: tuple[str, ...] = CLASS_LABELS,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Noise robustness evaluation requires held-out labelled audio.")

    conditions = (
        ("clean", None, False),
        ("white_noise_20db", 20, False),
        ("white_noise_10db", 10, False),
        ("white_noise_0db", 0, False),
        ("white_noise_10db_wiener", 10, True),
    )
    outputs: dict[str, Any] = {}
    model.eval()
    for condition_index, (name, snr_db, apply_denoising) in enumerate(conditions):
        predictions: list[int] = []
        confidences: list[float] = []
        truth: list[int] = []
        with torch.inference_mode():
            for sample_index, (path, label, _) in enumerate(records):
                decoded = decode_wav(path.read_bytes())
                samples = decoded.samples
                if snr_db is not None:
                    samples = add_white_noise(samples, snr_db, seed + sample_index)
                if apply_denoising:
                    samples = denoise_wiener(samples)
                feature = _model_features(samples)
                logits = model(
                    torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(device)
                )
                scores = F.softmax(logits, dim=1)[0].cpu().numpy()
                predictions.append(int(np.argmax(scores)))
                confidences.append(float(np.max(scores)))
                truth.append(label)

        truth_array = np.asarray(truth)
        predictions_array = np.asarray(predictions)
        outputs[name] = {
            "condition": name,
            "snr_db": snr_db,
            "denoising": "scipy.signal.wiener" if apply_denoising else "none",
            "sample_count": len(truth),
            "accuracy": float(accuracy_score(truth_array, predictions_array)),
            "macro_precision": float(
                precision_score(truth_array, predictions_array, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(truth_array, predictions_array, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(truth_array, predictions_array, average="macro", zero_division=0)
            ),
            "mean_raw_softmax_score": float(np.mean(confidences)),
        }
        if apply_denoising:
            outputs[name]["caveat"] = (
                "A simple Wiener filter baseline; effectiveness depends on audio and is not guaranteed."
            )

    return {
        "status": "evaluated",
        "seed": seed,
        "split": "speaker-independent held-out test speakers",
        "class_labels": list(class_labels),
        "conditions": outputs,
        "note": "Synthetic noise is added to copies of held-out test audio; original recordings are unchanged.",
    }


def calibration_report(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> dict[str, Any]:
    if probabilities.ndim != 2 or labels.ndim != 1 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Calibration inputs must have one class-score row per label.")
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    reliability_bins: list[dict[str, float | int]] = []
    expected_error = 0.0
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        count = int(np.count_nonzero(selected))
        if not count:
            continue
        mean_confidence = float(np.mean(confidence[selected]))
        accuracy = float(np.mean(correct[selected]))
        expected_error += count / len(labels) * abs(mean_confidence - accuracy)
        reliability_bins.append(
            {
                "lower_confidence": float(edges[index]),
                "upper_confidence": float(edges[index + 1]),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "sample_count": count,
            }
        )
    thresholds = [*np.linspace(0.0, 1.0, 21), 1.01]
    coverage_risk = []
    for threshold in thresholds:
        selected = confidence >= threshold
        if not selected.any():
            continue
        coverage_risk.append(
            {
                "threshold": threshold,
                "coverage": float(np.mean(selected)),
                "selective_risk": float(1.0 - np.mean(correct[selected])),
            }
        )
    return {
        "expected_calibration_error": float(expected_error),
        "bin_count": bins,
        "reliability_bins": reliability_bins,
        "coverage_risk": coverage_risk,
        "note": "Measured on held-out test data; softmax scores are not calibrated by this measurement.",
    }
