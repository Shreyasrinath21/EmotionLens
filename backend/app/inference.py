from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from requests.exceptions import RequestException

try:
    import torch
    from torch.nn import functional as F
    from .models import create_model
except ImportError:  # Vercel uses ONNX Runtime to keep the function under its size limit.
    torch = None
    F = None
    create_model = None

from .audio import (
    AudioData,
    detect_voice_activity,
    denoise_wiener,
    extract_features,
    segment_audio,
    TARGET_SAMPLE_RATE,
)
from .model_config import CLASS_LABELS, CREMA_D_CLASS_LABELS, MODEL_ARCHITECTURES


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "model.pt"
MODEL_INFO_PATH = ARTIFACTS_DIR / "model.json"
EVALUATION_PATH = ARTIFACTS_DIR / "evaluation.json"
PRETRAINED_ARCHITECTURE = "superb_wav2vec2_base_er"
PRETRAINED_MODEL_ID = "superb/wav2vec2-base-superb-er"
PRETRAINED_MODEL_REVISION = "441a7599c3b22107314dcbd9166621c5c83f2cc5"
PRETRAINED_CLASS_LABELS = ("neutral", "happy", "angry", "sad")
PRETRAINED_LABEL_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
}
PRETRAINED_MODEL_SIZE_BYTES = 756_665_603
CREMA_D_FILENAME = re.compile(
    r"\d{4}_[A-Z]{3}_(ANG|DIS|FEA|HAP|NEU|SAD)_(?:LO|MD|HI|XX)\.wav",
    re.IGNORECASE,
)
CREMA_D_FILENAME_LABELS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def _crema_d_filename_annotation(filename: str | None) -> dict[str, str] | None:
    if not filename:
        return None
    match = CREMA_D_FILENAME.fullmatch(Path(filename).name)
    if match is None:
        return None
    code = match.group(1).upper()
    return {
        "label": CREMA_D_FILENAME_LABELS[code],
        "source": "CREMA-D filename emotion code",
        "note": "Dataset annotation from the filename; not an EmotionLens model prediction.",
    }


class InferenceService:
    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.models: dict[str, torch.nn.Module] = {}
        self.architecture: str | None = None
        self.threshold: float | None = None
        self.model_version: str | None = None
        self.class_labels: tuple[str, ...] = CLASS_LABELS
        self.dataset: str | None = None
        self.pretrained_model: torch.nn.Module | None = None
        self.pretrained_feature_extractor: Any | None = None
        self.pretrained_load_lock = threading.Lock()
        self.device = torch.device("cpu") if torch is not None else None
        self.load_error: str | None = None
        self.load_checkpoint()

    def load_checkpoint(self) -> None:
        self.model = None
        self.models.clear()
        self.architecture = None
        self.threshold = None
        self.model_version = None
        self.class_labels = CLASS_LABELS
        self.dataset = None
        self.load_error = None
        if not MODEL_INFO_PATH.is_file() or (torch is not None and not CHECKPOINT_PATH.is_file()):
            return
        try:
            metadata = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
            architecture = metadata["architecture"]
            class_labels = tuple(metadata["class_labels"])
            supported_label_sets = (CLASS_LABELS, CREMA_D_CLASS_LABELS)
            if class_labels not in supported_label_sets:
                raise ValueError("Checkpoint class mapping does not match a supported dataset.")
            if torch is None:
                model = self._open_onnx_session(architecture)
            else:
                model = create_model(architecture, len(class_labels))
                state = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                model.eval()
            threshold = float(metadata["uncertainty_threshold"])
            if not 0 <= threshold <= 1.01:
                raise ValueError("The checkpoint uncertainty threshold must be in [0, 1.01].")
            self.model = model
            self.models[architecture] = model
            self.architecture = architecture
            self.class_labels = class_labels
            self.dataset = str(metadata.get("dataset", ""))
            self.threshold = threshold
            self.model_version = str(metadata["model_version"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
            self.load_error = f"Model checkpoint could not be loaded: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def pretrained_available(self) -> bool:
        return torch is not None

    def model_info(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if MODEL_INFO_PATH.is_file():
            try:
                metadata = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
        return {
            **metadata,
            "available": self.available,
            "load_error": self.load_error,
            "class_labels": list(self.class_labels),
            "dataset": self.dataset or metadata.get("dataset"),
            "architecture": self.architecture or metadata.get("architecture"),
            "available_architectures": self.available_architectures(),
            "pretrained_model": {
                "model_id": PRETRAINED_MODEL_ID,
                "revision": PRETRAINED_MODEL_REVISION,
                "loaded": self.pretrained_model is not None,
                "class_labels": list(PRETRAINED_CLASS_LABELS),
                "dataset": "IEMOCAP via SUPERB",
                "license": "Apache-2.0 on Hugging Face model card; IEMOCAP source-data terms still apply",
                "external_metric": {
                    "name": "SUPERB session-1 accuracy",
                    "value": 0.6258,
                    "note": "Reported externally; not measured by EmotionLens and not comparable to its RAVDESS results.",
                },
                "download_size_bytes": PRETRAINED_MODEL_SIZE_BYTES,
                "download_on_first_use": torch is not None,
                "available": torch is not None,
                "local_audio_inference": True,
            },
            "model_version": self.model_version or metadata.get("model_version"),
            "uncertainty_threshold": self.threshold
            if self.threshold is not None
            else metadata.get("uncertainty_threshold"),
            "note": None
            if self.available
            else "No local trained checkpoint is installed. A separately labelled pretrained model may be available on supported runtimes.",
        }

    def available_architectures(self) -> list[str]:
        available = [self.architecture] if self.available and self.architecture else []
        if torch is not None and PRETRAINED_ARCHITECTURE not in available:
            available.append(PRETRAINED_ARCHITECTURE)
        comparison = self.artifact_report("comparison")
        if comparison.get("status") != "evaluated":
            return available
        for architecture in comparison.get("models", {}):
            if architecture not in MODEL_ARCHITECTURES or architecture in available:
                continue
            model_dir = ARTIFACTS_DIR / "models" / architecture
            checkpoint = model_dir / ("model.onnx" if torch is None else "model.pt")
            if checkpoint.is_file() and (model_dir / "model.json").is_file():
                available.append(architecture)
        return available

    def _open_onnx_session(self, architecture: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("The ONNX Runtime dependency is missing.") from exc
        model_path = ARTIFACTS_DIR / "models" / architecture / "model.onnx"
        if not model_path.is_file():
            raise RuntimeError(f"The ONNX model for {architecture} is not available.")
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def _load_pretrained_model(self) -> tuple[Any, Any]:
        if torch is None:
            raise RuntimeError("The optional pretrained model is unavailable in the compact Vercel deployment.")
        if self.pretrained_model is not None and self.pretrained_feature_extractor is not None:
            return self.pretrained_model, self.pretrained_feature_extractor
        with self.pretrained_load_lock:
            if self.pretrained_model is not None and self.pretrained_feature_extractor is not None:
                return self.pretrained_model, self.pretrained_feature_extractor
            try:
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            except ImportError as exc:
                raise RuntimeError(
                    "The pretrained model dependency is missing. Install backend requirements and restart."
                ) from exc

            cache_dir = (
                Path(tempfile.gettempdir()) / "emotionlens-pretrained-cache"
                if os.environ.get("VERCEL")
                else ARTIFACTS_DIR / "pretrained_cache"
            )
            try:
                feature_extractor = AutoFeatureExtractor.from_pretrained(
                    PRETRAINED_MODEL_ID,
                    revision=PRETRAINED_MODEL_REVISION,
                    cache_dir=cache_dir,
                )
                model = AutoModelForAudioClassification.from_pretrained(
                    PRETRAINED_MODEL_ID,
                    revision=PRETRAINED_MODEL_REVISION,
                    cache_dir=cache_dir,
                )
                id_to_label = model.config.id2label
                labels = tuple(
                    PRETRAINED_LABEL_MAP.get(
                        str(id_to_label.get(index, id_to_label.get(str(index), ""))).lower(),
                        "",
                    )
                    for index in range(len(PRETRAINED_CLASS_LABELS))
                )
                if labels != PRETRAINED_CLASS_LABELS:
                    raise ValueError(
                        f"The downloaded model has an unsupported class mapping: {labels!r}."
                    )
                model.to(self.device)
                model.eval()
            except (OSError, RequestException, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Could not load {PRETRAINED_MODEL_ID} locally. Check the network connection and available disk space, then retry. Details: {exc}"
                ) from exc
            self.pretrained_model = model
            self.pretrained_feature_extractor = feature_extractor
            return model, feature_extractor

    def _resolve_model(
        self, architecture: str | None
    ) -> tuple[Any, float, str, str, tuple[str, ...], str]:
        selected = architecture or self.architecture
        if selected == self.architecture and self.model is not None:
            return (
                self.model,
                self.threshold,
                self.model_version,
                selected,
                self.class_labels,
                self.dataset or "",
            )
        if selected is None or selected not in MODEL_ARCHITECTURES:
            raise RuntimeError("Select an available supported model checkpoint.")
        model_dir = ARTIFACTS_DIR / "models" / selected
        try:
            metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"The saved {selected} model configuration is unavailable or invalid: {exc}"
            ) from exc
        if metadata.get("architecture") != selected:
            raise RuntimeError(f"The saved {selected} model metadata has an invalid architecture.")
        class_labels = tuple(metadata.get("class_labels", ()))
        if class_labels not in (CLASS_LABELS, CREMA_D_CLASS_LABELS):
            raise RuntimeError(f"The saved {selected} model uses an unsupported class mapping.")
        model = self.models.get(selected)
        if model is None:
            if torch is None:
                model = self._open_onnx_session(selected)
            else:
                model = create_model(selected, len(class_labels))
                state = torch.load(
                    model_dir / "model.pt",
                    map_location=self.device,
                    weights_only=True,
                )
                model.load_state_dict(state)
                model.eval()
            self.models[selected] = model
        try:
            threshold = float(metadata["uncertainty_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"The saved {selected} uncertainty threshold is invalid.") from exc
        if not 0 <= threshold <= 1.01:
            raise RuntimeError(f"The saved {selected} uncertainty threshold is invalid.")
        return (
            model,
            threshold,
            str(metadata["model_version"]),
            selected,
            class_labels,
            str(metadata.get("dataset", "")),
        )

    def evaluation_results(self) -> dict[str, Any]:
        if not EVALUATION_PATH.is_file():
            return {
                "status": "not_evaluated",
                "message": "No held-out evaluation report is available yet.",
            }
        try:
            return json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "unavailable",
                "message": f"Evaluation report could not be read: {exc}",
            }

    def artifact_report(self, name: str) -> dict[str, Any]:
        paths = {
            "comparison": ARTIFACTS_DIR / "comparison.json",
            "robustness": ARTIFACTS_DIR / "robustness.json",
            "cross_corpus": ARTIFACTS_DIR / "cross_corpus.json",
            "dataset_audit": ARTIFACTS_DIR / "dataset_audit.json",
        }
        try:
            path = paths[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported artifact report: {name}") from exc
        if not path.is_file():
            messages = {
                "cross_corpus": (
                    "No cross-corpus report exists. Obtain permitted target audio and run "
                    "python -m app.cross_corpus --dataset ..\\datasets\\CREMA-D "
                    "--dataset-format crema-d."
                ),
                "dataset_audit": (
                    "No dataset audit exists. Run "
                    "python -m app.training --dataset ..\\datasets\\RAVDESS "
                    "--dataset-format ravdess --audit-dataset."
                ),
            }
            return {
                "status": "not_evaluated",
                "message": messages.get(name, f"No {name} experiment has been run."),
            }
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "unavailable",
                "message": f"{name.capitalize()} report could not be read: {exc}",
            }

    def analyze_features(
        self, audio: AudioData, denoise: bool = False
    ) -> dict[str, Any]:
        return self._analyze_features(audio, denoise=denoise)

    def _prepare_audio(self, audio: AudioData, denoise: bool) -> AudioData:
        if not denoise:
            return audio
        return AudioData(
            samples=denoise_wiener(audio.samples),
            sample_rate=audio.sample_rate,
            original_sample_rate=audio.original_sample_rate,
            original_channels=audio.original_channels,
            original_duration=audio.original_duration,
        )

    def _analyze_features(self, audio: AudioData, denoise: bool) -> dict[str, Any]:
        audio = self._prepare_audio(audio, denoise)
        return {
            "status": "features_only",
            "audio": {
                "duration_seconds": audio.original_duration,
                "sample_rate_hz": audio.original_sample_rate,
                "analysis_sample_rate_hz": audio.sample_rate,
                "channels": audio.original_channels,
                "preprocessing": self._preprocessing_description(denoise),
            },
            "features": extract_features(audio.samples),
            "note": "Acoustic measurements only; no emotion prediction was requested.",
        }

    @staticmethod
    def _preprocessing_description(denoise: bool) -> str:
        steps = "Mono mixdown, 16 kHz resampling, peak normalization"
        if denoise:
            steps += ", experimental Wiener denoising"
        return steps

    def analyze(
        self,
        audio: AudioData,
        include_attribution: bool = False,
        architecture: str | None = None,
        input_filename: str | None = None,
        denoise: bool = False,
    ) -> dict[str, Any]:
        audio = self._prepare_audio(audio, denoise)
        audio_metadata = {
            "duration_seconds": audio.original_duration,
            "sample_rate_hz": audio.original_sample_rate,
            "analysis_sample_rate_hz": audio.sample_rate,
            "channels": audio.original_channels,
            "preprocessing": self._preprocessing_description(denoise),
        }
        activity = detect_voice_activity(audio.samples)
        if not activity["voice_detected"]:
            warning = (
                "No likely active-audio interval was detected. This is an energy-based "
                "silence check, not semantic speech recognition; no emotion prediction was made."
            )
            return {
                "status": "features_only",
                "analysis_mode": "no_voice_activity",
                "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
                "summary": None,
                "segments": [],
                "audio": audio_metadata,
                "features": extract_features(audio.samples),
                "source_corpus_hint": None,
                "filename_annotation": None,
                "analysis_warning": warning,
                "reliability": {
                    "model_available": self.available,
                    "model_version": self.model_version,
                    "threshold": None,
                    "message": warning,
                },
                "caveat": activity["limitation"],
            }
        filename_annotation = _crema_d_filename_annotation(input_filename)
        crema_d_filename = filename_annotation is not None
        if architecture is None and (self.model is None or self.threshold is None):
            warning = (
                "This filename follows the CREMA-D naming pattern, but no compatible "
                "CREMA-D model is installed. No cross-corpus emotion prediction was made."
                if crema_d_filename
                else "Audio features were measured, but emotion classification is unavailable until a trained model checkpoint is installed."
            )
            return {
                "status": "features_only",
                "analysis_mode": (
                    "unsupported_corpus_features_only"
                    if crema_d_filename
                    else "audio_features_only"
                ),
                "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
                "summary": None,
                "segments": [],
                "audio": audio_metadata,
                "features": extract_features(audio.samples),
                "source_corpus_hint": "CREMA-D" if crema_d_filename else None,
                "filename_annotation": filename_annotation,
                "analysis_warning": warning if crema_d_filename else None,
                "reliability": {
                    "model_available": self.available,
                    "model_version": None,
                    "threshold": None,
                    "message": warning,
                },
                "caveat": (
                    "Acoustic features describe signal properties only; they do not identify a speaker's emotion."
                ),
            }
        if architecture == PRETRAINED_ARCHITECTURE:
            warning = (
                "This file follows the CREMA-D naming pattern. The selected pretrained "
                "model was trained on IEMOCAP and has not been evaluated on CREMA-D, so "
                "no cross-corpus emotion prediction was made."
            )
            if crema_d_filename:
                return {
                    "status": "features_only",
                    "analysis_mode": "unsupported_corpus_features_only",
                    "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": None,
                    "segments": [],
                    "audio": audio_metadata,
                    "features": extract_features(audio.samples),
                    "source_corpus_hint": "CREMA-D",
                    "filename_annotation": filename_annotation,
                    "analysis_warning": warning,
                    "reliability": {
                        "model_available": True,
                        "model_version": f"{PRETRAINED_MODEL_ID}@{PRETRAINED_MODEL_REVISION}",
                        "threshold": None,
                        "message": warning,
                    },
                    "caveat": "Acoustic features describe signal properties only; they do not identify a speaker's emotion.",
                }
            return self._predict_pretrained_audio(audio, audio_metadata)
        try:
            (
                model,
                threshold,
                model_version,
                selected_architecture,
                class_labels,
                dataset,
            ) = self._resolve_model(architecture)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"The {architecture} checkpoint is not available. Retrain with --compare."
            ) from exc
        if crema_d_filename and dataset.casefold() != "crema-d":
            warning = (
                "This filename follows the CREMA-D naming pattern, but the selected model "
                f"was trained on {dataset or 'a different dataset'}. No cross-corpus emotion "
                "prediction was made. Train and evaluate a CREMA-D model before analysing "
                "these recordings."
            )
            return {
                "status": "features_only",
                "analysis_mode": "unsupported_corpus_features_only",
                "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
                "summary": None,
                "segments": [],
                "audio": audio_metadata,
                "features": extract_features(audio.samples),
                "source_corpus_hint": "CREMA-D",
                "filename_annotation": filename_annotation,
                "analysis_warning": warning,
                "reliability": {
                    "model_available": self.available,
                    "model_version": model_version,
                    "threshold": None,
                    "message": warning,
                },
                "caveat": (
                    "Acoustic features describe signal properties only; they do not identify "
                    "a speaker's emotion."
                ),
            }
        return self._predict_audio(
            audio,
            audio_metadata,
            include_attribution,
            model,
            threshold,
            model_version,
            selected_architecture,
            class_labels,
        )

    def _predict_pretrained_audio(
        self,
        audio: AudioData,
        audio_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        model, feature_extractor = self._load_pretrained_model()
        segment_results: list[dict[str, Any]] = []
        aggregate_scores = np.zeros(len(PRETRAINED_CLASS_LABELS), dtype=np.float64)
        segments = segment_audio(audio.samples)
        with torch.inference_mode():
            for index, (start, end, _) in enumerate(segments):
                start_sample = int(round(start * audio.sample_rate))
                end_sample = int(round(end * audio.sample_rate))
                waveform = audio.samples[start_sample:end_sample]
                model_inputs = feature_extractor(
                    waveform,
                    sampling_rate=TARGET_SAMPLE_RATE,
                    padding=True,
                    return_tensors="pt",
                )
                logits = model(
                    **{
                        key: value.to(self.device)
                        for key, value in model_inputs.items()
                    }
                ).logits
                if logits.ndim != 2 or logits.shape != (1, len(PRETRAINED_CLASS_LABELS)):
                    raise RuntimeError("The pretrained model returned an invalid class-score shape.")
                scores = F.softmax(logits, dim=1)[0].cpu().numpy()
                if not np.isfinite(scores).all():
                    raise RuntimeError("The pretrained model returned non-finite class scores.")
                aggregate_scores += scores
                top_index = int(np.argmax(scores))
                segment_results.append(
                    {
                        "index": index,
                        "start_seconds": round(start, 3),
                        "end_seconds": round(end, 3),
                        "label": PRETRAINED_CLASS_LABELS[top_index],
                        "candidate_label": PRETRAINED_CLASS_LABELS[top_index],
                        "score": float(scores[top_index]),
                        "scores": {
                            label: float(score)
                            for label, score in zip(PRETRAINED_CLASS_LABELS, scores, strict=True)
                        },
                    }
                )

        aggregate_scores /= len(segment_results)
        top_index = int(np.argmax(aggregate_scores))
        top_label = PRETRAINED_CLASS_LABELS[top_index]
        model_version = f"{PRETRAINED_MODEL_ID}@{PRETRAINED_MODEL_REVISION}"
        return {
            "status": "complete",
            "analysis_mode": PRETRAINED_ARCHITECTURE,
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "label": top_label,
                "candidate_label": top_label,
                "top_score": float(aggregate_scores[top_index]),
                "uncertain": False,
                "scores": {
                    label: float(score)
                    for label, score in zip(PRETRAINED_CLASS_LABELS, aggregate_scores, strict=True)
                },
            },
            "segments": segment_results,
            "audio": audio_metadata,
            "features": extract_features(audio.samples),
            "explainability": {
                "status": "unavailable",
                "method": None,
                "target_class": None,
                "feature_sensitivity": None,
                "limitation": "Input-gradient sensitivity is not implemented for the raw-waveform pretrained model.",
            },
            "reliability": {
                "threshold": None,
                "score_description": "Raw softmax scores are not calibrated; no local rejection threshold has been evaluated.",
                "model_version": model_version,
                "architecture": PRETRAINED_ARCHITECTURE,
                "dataset": "IEMOCAP via SUPERB",
                "segment_count": len(segment_results),
                "uncertain_segment_count": 0,
                "out_of_distribution_detection": "not implemented",
                "external_metric": "SUPERB reports 62.58% session-1 accuracy; this is not an EmotionLens evaluation.",
            },
            "caveat": (
                "English IEMOCAP-trained four-class research model. It is not validated on "
                "RAVDESS or everyday recordings; its scores are not calibrated or facts about a speaker."
            ),
        }

    @staticmethod
    def _scores_for_log_mel(model: Any, log_mel: np.ndarray) -> np.ndarray:
        if torch is None:
            input_name = model.get_inputs()[0].name
            logits = model.run(
                None,
                {input_name: np.asarray(log_mel, dtype=np.float32)[None, None, :, :]},
            )[0][0]
            shifted = logits - np.max(logits)
            probabilities = np.exp(shifted)
            return probabilities / probabilities.sum()
        tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)
        logits = model(tensor)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError("The model returned an invalid class-score shape.")
        return F.softmax(logits, dim=1)[0].cpu().numpy()

    def _predict_audio(
        self,
        audio: AudioData,
        audio_metadata: dict[str, Any],
        include_attribution: bool,
        model: Any,
        threshold: float,
        model_version: str,
        architecture: str,
        class_labels: tuple[str, ...],
    ) -> dict[str, Any]:
        overall_features = extract_features(audio.samples)
        segments = segment_audio(audio.samples)
        segment_results: list[dict[str, Any]] = []
        aggregate_scores = np.zeros(len(class_labels), dtype=np.float64)
        with torch.inference_mode() if torch is not None else nullcontext():
            for index, (start, end, log_mel) in enumerate(segments):
                scores = self._scores_for_log_mel(model, log_mel)
                if scores.ndim != 1 or scores.shape != (len(class_labels),):
                    raise RuntimeError("The model returned an invalid class-score shape.")
                if not np.isfinite(scores).all():
                    raise RuntimeError("The model returned non-finite class scores.")
                aggregate_scores += scores
                top_index = int(np.argmax(scores))
                segment_results.append(
                    {
                        "index": index,
                        "start_seconds": round(start, 3),
                        "end_seconds": round(end, 3),
                        "label": class_labels[top_index]
                        if float(scores[top_index]) >= threshold
                        else "uncertain",
                        "candidate_label": class_labels[top_index],
                        "score": float(scores[top_index]),
                        "scores": {
                            label: float(score)
                            for label, score in zip(class_labels, scores, strict=True)
                        },
                    }
                )

        aggregate_scores /= len(segment_results)
        top_index = int(np.argmax(aggregate_scores))
        top_score = float(aggregate_scores[top_index])
        uncertain = top_score < threshold
        explainability = {
            "status": "not_requested",
            "method": None,
            "target_class": None,
            "feature_sensitivity": None,
            "limitation": (
                "Input gradients indicate model sensitivity, not causal evidence or a person's emotion."
            ),
        }
        if include_attribution and torch is None:
            explainability["status"] = "unavailable"
            explainability["limitation"] = "Input-gradient maps are unavailable in the compact ONNX deployment."
        elif include_attribution:
            sensitivity = np.zeros((64, 300), dtype=np.float32)
            with torch.enable_grad():
                for _, _, log_mel in segments:
                    tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)
                    tensor.requires_grad_(True)
                    logits = model(tensor)
                    target_score = F.softmax(logits, dim=1)[:, top_index].sum()
                    gradient = torch.autograd.grad(target_score, tensor)[0]
                    sensitivity += gradient.detach().abs().squeeze(0).squeeze(0).cpu().numpy()
            maximum = float(np.max(sensitivity))
            if maximum > 0:
                sensitivity /= maximum
            explainability = {
                **explainability,
                "status": "available",
                "method": "Mean absolute input gradient of the candidate class softmax score",
                "target_class": class_labels[top_index],
                "feature_sensitivity": np.round(sensitivity, 4).tolist(),
            }
        return {
            "status": "complete",
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "label": "uncertain" if uncertain else class_labels[top_index],
                "candidate_label": class_labels[top_index],
                "top_score": top_score,
                "uncertain": uncertain,
                "scores": {
                    label: float(score)
                    for label, score in zip(class_labels, aggregate_scores, strict=True)
                },
            },
            "segments": segment_results,
            "audio": audio_metadata,
            "features": overall_features,
            "explainability": explainability,
            "reliability": {
                "threshold": threshold,
                "score_description": "Raw softmax scores; calibration has not been established.",
                "model_version": model_version,
                "architecture": architecture,
                "window_seconds": 3,
                "hop_seconds": 2,
                "segment_count": len(segment_results),
                "uncertain_segment_count": sum(
                    result["label"] == "uncertain" for result in segment_results
                ),
                "out_of_distribution_detection": "not implemented",
            },
            "caveat": "Model estimates are not objective facts about a speaker's internal state.",
        }
