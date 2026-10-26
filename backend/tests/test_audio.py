from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

from app.audio import (
    AudioData,
    AudioInputError,
    decode_wav,
    detect_voice_activity,
    extract_features,
    segment_audio,
)
from app.inference import InferenceService
from app.inference import PRETRAINED_ARCHITECTURE, PRETRAINED_CLASS_LABELS
from app.models import (
    CLASS_LABELS,
    CREMA_D_CLASS_LABELS,
    EmotionCNN,
    EmotionCNNBiLSTMAttention,
    EmotionCNNLSTM,
    create_model,
)
from app.experiments import add_white_noise, calibration_report, denoise_wiener
from app.training import (
    _choose_threshold,
    _read_crema_d,
    _read_ravdess,
    _speaker_splits,
    audit_dataset,
    evaluate_cross_corpus,
)


def wav_bytes(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    output = BytesIO()
    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def test_silent_wav_is_rejected() -> None:
    with pytest.raises(AudioInputError, match="silent"):
        decode_wav(wav_bytes(np.zeros(16_000, dtype=np.float32)))


def test_invalid_wav_is_rejected() -> None:
    with pytest.raises(AudioInputError):
        decode_wav(b"not audio")


def test_non_wav_audio_is_rejected_even_if_renamed() -> None:
    encoded = BytesIO()
    sf.write(encoded, np.ones(16_000, dtype=np.float32) * 0.1, 16_000, format="FLAC")
    with pytest.raises(AudioInputError, match="Only WAV"):
        decode_wav(encoded.getvalue())


def test_stereo_is_mixed_to_mono_and_resampled() -> None:
    samples = np.column_stack(
        (
            np.sin(np.arange(24_000) * 0.03),
            np.sin(np.arange(24_000) * 0.04),
        )
    ).astype(np.float32)
    decoded = decode_wav(wav_bytes(samples, sample_rate=24_000))
    assert decoded.original_channels == 2
    assert decoded.sample_rate == 16_000
    assert decoded.samples.ndim == 1
    assert decoded.original_duration == pytest.approx(1.0)


def test_too_long_audio_is_rejected() -> None:
    samples = np.ones(16_000 * 31, dtype=np.float32) * 0.1
    with pytest.raises(AudioInputError, match="30 seconds"):
        decode_wav(wav_bytes(samples))


def test_threshold_applies_to_actual_inference_result() -> None:
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            peak = inputs.flatten(1).max(dim=1).values
            return torch.stack(
                [
                    peak * 4.0 + 4.0,
                    peak * 0.0,
                    peak * 0.0,
                    peak * 0.0,
                    peak * 0.0,
                    peak * 0.0,
                    peak * 0.0,
                    peak * 0.0,
                ],
                dim=1,
            )

    service = InferenceService()
    service.model = ConstantModel()
    service.threshold = 1.01
    service.architecture = "test"
    service.model_version = "test-model"
    samples = np.sin(np.arange(16_000) * 0.03).astype(np.float32)
    audio = AudioData(samples, 16_000, 16_000, 1, 1.0)

    rejected = service.analyze(audio)
    assert rejected["summary"]["uncertain"] is True
    assert rejected["segments"][0]["label"] == "uncertain"

    service.threshold = 0.0
    accepted = service.analyze(audio)
    assert accepted["summary"]["uncertain"] is False
    assert accepted["summary"]["label"] == "neutral"
    explained = service.analyze(audio, include_attribution=True)
    assert explained["explainability"]["status"] == "available"
    assert explained["explainability"]["target_class"] == "neutral"
    assert len(explained["explainability"]["feature_sensitivity"]) == 64


def test_pretrained_model_returns_per_audio_four_class_scores(monkeypatch) -> None:
    class FakeFeatureExtractor:
        def __call__(self, waveform, **kwargs):
            return {"input_values": torch.from_numpy(np.asarray(waveform).copy()).unsqueeze(0)}

    class FakePretrainedModel(torch.nn.Module):
        def forward(self, input_values: torch.Tensor) -> Any:
            mean = input_values.mean()
            return type("Output", (), {
                "logits": torch.stack((mean * 5, -mean * 5, mean * 0, mean * 0)).unsqueeze(0)
            })()

    service = InferenceService()
    monkeypatch.setattr(
        service,
        "_load_pretrained_model",
        lambda: (FakePretrainedModel(), FakeFeatureExtractor()),
    )
    positive_audio = AudioData(
        np.full(16_000, 0.2, dtype=np.float32), 16_000, 16_000, 1, 1.0
    )
    negative_audio = AudioData(
        np.full(16_000, -0.2, dtype=np.float32), 16_000, 16_000, 1, 1.0
    )

    positive = service.analyze(positive_audio, architecture=PRETRAINED_ARCHITECTURE)
    negative = service.analyze(negative_audio, architecture=PRETRAINED_ARCHITECTURE)

    assert tuple(positive["summary"]["scores"]) == PRETRAINED_CLASS_LABELS
    assert positive["summary"]["label"] == "neutral"
    assert negative["summary"]["label"] == "happy"
    assert positive["summary"]["scores"] != negative["summary"]["scores"]
    assert positive["reliability"]["threshold"] is None
    assert positive["reliability"]["dataset"] == "IEMOCAP via SUPERB"
    assert positive["explainability"]["status"] == "unavailable"


def test_pretrained_selector_is_available_without_local_checkpoint() -> None:
    service = InferenceService()
    assert PRETRAINED_ARCHITECTURE in service.available_architectures()


def test_very_short_audio_is_rejected() -> None:
    with pytest.raises(AudioInputError, match="0.05 seconds"):
        decode_wav(wav_bytes(np.ones(400, dtype=np.float32) * 0.1))


def test_features_and_segments_have_expected_dimensions() -> None:
    samples = np.sin(np.arange(16_000 * 4) * 0.04).astype(np.float32)
    features = extract_features(samples)
    segments = segment_audio(samples)
    assert len(features["log_mel"]) == 64
    assert len(features["mfcc"]) == 13
    assert len(segments) == 2
    assert segments[0][0] == 0
    assert segments[1][0] == 2
    assert segments[-1][1] == pytest.approx(4)
    assert features["voice_activity"]["voice_detected"] is True


def test_energy_activity_estimator_finds_intervals_and_silence() -> None:
    samples = np.zeros(16_000 * 3, dtype=np.float32)
    samples[16_000 : 32_000] = 0.4 * np.sin(
        np.arange(16_000, dtype=np.float32) * 0.04
    )
    detected = detect_voice_activity(samples)
    silent = detect_voice_activity(np.zeros(16_000, dtype=np.float32))
    assert detected["voice_detected"] is True
    assert detected["active_intervals_seconds"][0][0] < 1.0
    assert detected["active_intervals_seconds"][0][1] > 2.0
    assert detected["estimated_active_fraction"] < 0.6
    assert silent["voice_detected"] is False


def test_inference_returns_no_prediction_when_activity_is_not_detected() -> None:
    service = InferenceService()
    service.model = None
    service.threshold = None
    audio = AudioData(np.zeros(16_000, dtype=np.float32), 16_000, 16_000, 1, 1.0)
    result = service.analyze(audio)
    assert result["status"] == "features_only"
    assert result["analysis_mode"] == "no_voice_activity"
    assert result["summary"] is None


def test_speaker_splits_have_no_overlap() -> None:
    records = [(Path(f"{speaker}.wav"), speaker % 8, str(speaker)) for speaker in range(12)]
    splits = _speaker_splits(records, seed=42)
    speaker_sets = {name: {record[2] for record in rows} for name, rows in splits.items()}
    assert not speaker_sets["train"] & speaker_sets["validation"]
    assert not speaker_sets["train"] & speaker_sets["test"]
    assert not speaker_sets["validation"] & speaker_sets["test"]
    assert set.union(*speaker_sets.values()) == {str(speaker) for speaker in range(12)}


def test_dataset_audit_reports_real_wav_and_split_counts(tmp_path: Path) -> None:
    for actor in range(1, 4):
        filename = f"03-01-0{actor}-01-01-01-{actor:02d}.wav"
        samples = 0.2 * np.sin(np.arange(8_000, dtype=np.float32) * 0.04)
        (tmp_path / filename).write_bytes(wav_bytes(samples))
    (tmp_path / "unlabelled.wav").write_bytes(wav_bytes(np.ones(8_000, dtype=np.float32) * 0.1))

    report = audit_dataset(tmp_path, "ravdess")

    assert report["status"] == "audited"
    assert report["total_wav_files"] == 4
    assert report["recognized_labelled_files"] == 3
    assert report["valid_labelled_files"] == 3
    assert report["unrecognized_or_unlabelled_wav_count"] == 1
    assert report["speaker_count"] == 3
    assert report["speaker_split_audit_seed_42"] is not None


def test_ravdess_parser_accepts_actor_folder_and_numeric_filename(tmp_path: Path) -> None:
    actor_dir = tmp_path / "Actor_01"
    actor_dir.mkdir()
    recording = actor_dir / "03-01-03-01-01-01-01.wav"
    recording.write_bytes(b"placeholder wav content for filename parsing")

    records = _read_ravdess(tmp_path)

    assert records == [(recording, 2, "01")]


def test_ravdess_parser_ignores_non_speech_modality(tmp_path: Path) -> None:
    actor_dir = tmp_path / "Actor_01"
    actor_dir.mkdir()
    (actor_dir / "01-01-03-01-01-01-01.wav").touch()
    with pytest.raises(ValueError, match="audio-only speech"):
        _read_ravdess(tmp_path)


def test_crema_d_parser_maps_emotion_codes_and_actor_ids(tmp_path: Path) -> None:
    recordings = (
        ("1001_IEO_ANG_HI.wav", "angry"),
        ("1002_TAI_DIS_LO.wav", "disgust"),
        ("1003_IOM_FEA_MD.wav", "fearful"),
        ("1004_TIE_HAP_XX.wav", "happy"),
        ("1005_IWL_NEU_HI.wav", "neutral"),
        ("1006_TSI_SAD_LO.wav", "sad"),
    )
    for filename, _ in recordings:
        (tmp_path / filename).touch()

    records = _read_crema_d(tmp_path)

    assert len(records) == len(recordings)
    assert {
        CREMA_D_CLASS_LABELS[index] for _, index, _ in records
    } == set(CREMA_D_CLASS_LABELS)
    assert {speaker for _, _, speaker in records} == {str(index) for index in range(1001, 1007)}


def test_crema_d_parser_ignores_invalid_filename_fields(tmp_path: Path) -> None:
    (tmp_path / "1001_IEO_ANG_EXTRA_HI.wav").touch()
    (tmp_path / "1002_IEO_UNKNOWN_HI.wav").touch()
    with pytest.raises(ValueError, match="CREMA-D WAV"):
        _read_crema_d(tmp_path)


def test_model_factory_supports_crema_d_six_class_head() -> None:
    for architecture in ("cnn", "cnn_lstm", "cnn_bilstm_attention"):
        logits = create_model(architecture, len(CREMA_D_CLASS_LABELS))(
            torch.zeros((1, 1, 64, 300))
        )
        assert logits.shape == (1, len(CREMA_D_CLASS_LABELS))


def test_cross_corpus_evaluation_uses_shared_labels_without_finetuning(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_model = create_model("cnn", len(CLASS_LABELS))
    torch.save(source_model.state_dict(), source_dir / "model.pt")
    (source_dir / "model.json").write_text(
        json.dumps(
            {
                "architecture": "cnn",
                "class_labels": list(CLASS_LABELS),
                "dataset": "RAVDESS",
                "model_version": "temporary-source-model",
                "uncertainty_threshold": 1.01,
            }
        ),
        encoding="utf-8",
    )
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    for index, code in enumerate(("NEU", "HAP", "SAD", "ANG", "FEA", "DIS"), 1001):
        samples = 0.2 * np.sin(np.arange(8_000, dtype=np.float32) * (0.03 + index * 1e-6))
        (target_dir / f"{index}_IEO_{code}_HI.wav").write_bytes(wav_bytes(samples))

    report = evaluate_cross_corpus(source_dir, target_dir, "crema-d")

    assert report["status"] == "evaluated"
    assert report["source_dataset"] == "RAVDESS"
    assert report["target_dataset"] == "CREMA-D"
    assert report["target_sample_count"] == 6
    assert report["shared_class_labels"] == ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
    assert report["confusion_matrix"] and len(report["confusion_matrix"]) == 6
    assert report["evaluation_protocol"].find("No target fine-tuning") >= 0


def test_crema_d_checkpoint_labels_flow_through_inference() -> None:
    service = InferenceService()
    service.model = create_model("cnn_lstm", len(CREMA_D_CLASS_LABELS)).eval()
    service.threshold = 0.0
    service.architecture = "cnn_lstm"
    service.class_labels = CREMA_D_CLASS_LABELS
    service.dataset = "CREMA-D"
    service.model_version = "test-crema-d"
    samples = np.sin(np.arange(16_000, dtype=np.float32) * 0.03)
    audio = AudioData(samples, 16_000, 16_000, 1, 1.0)

    result = service.analyze(audio, input_filename="1001_IEO_ANG_HI.wav")

    assert result["status"] == "complete"
    assert set(result["summary"]["scores"]) == set(CREMA_D_CLASS_LABELS)
    assert result["reliability"]["model_version"] == "test-crema-d"


def test_uncertainty_threshold_uses_validation_labels() -> None:
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.51, 0.49]])
    threshold = _choose_threshold(probabilities, np.array([0, 1, 0]))
    assert threshold > 0.5


def test_noise_generation_is_deterministic_and_denoising_is_finite() -> None:
    samples = np.sin(np.arange(16_000) * 0.05).astype(np.float32)
    first = add_white_noise(samples, snr_db=10, seed=7)
    second = add_white_noise(samples, snr_db=10, seed=7)
    assert np.array_equal(first, second)
    assert np.isfinite(denoise_wiener(first)).all()


def test_calibration_report_contains_reliability_and_coverage_risk() -> None:
    report = calibration_report(
        np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]]),
        np.array([0, 1, 0]),
        bins=5,
    )
    assert 0 <= report["expected_calibration_error"] <= 1
    assert report["reliability_bins"]
    assert report["coverage_risk"]


@pytest.mark.parametrize(
    "model_type", [EmotionCNN, EmotionCNNLSTM, EmotionCNNBiLSTMAttention]
)
def test_model_output_matches_class_mapping(model_type: type[torch.nn.Module]) -> None:
    model = model_type().eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 1, 64, 300))
    assert output.shape == (2, len(CLASS_LABELS))
