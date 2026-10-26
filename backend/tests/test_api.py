from __future__ import annotations

import asyncio
import math
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import soundfile as sf

from app.main import app, service
from app.inference import PRETRAINED_ARCHITECTURE


async def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


def request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    return asyncio.run(_request(method, path, **kwargs))


def test_health_reports_service_state() -> None:
    response = request("GET", "/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_endpoint_does_not_claim_checkpoint_when_missing() -> None:
    response = request("GET", "/api/model")
    assert response.status_code == 200
    assert isinstance(response.json()["available"], bool)


def test_feature_endpoint_returns_measurements_without_classifying() -> None:
    wav = BytesIO()
    sf.write(wav, [0.2] * 16_000, 16_000, format="WAV")
    response = request(
        "POST",
        "/api/features",
        files={"file": ("reference.wav", wav.getvalue())},
        data={"consent": "true"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "features_only"
    assert response.json()["features"]["acoustic"]["rms_mean"] is not None
    assert response.json()["features"]["voice_activity"]["voice_detected"] is True


def test_feature_endpoint_applies_requested_wiener_denoising() -> None:
    samples = [0.2 * math.sin(index * 0.04) for index in range(16_000)]
    wav = BytesIO()
    sf.write(wav, samples, 16_000, format="WAV")
    response = request(
        "POST",
        "/api/features",
        files={"file": ("reference.wav", wav.getvalue())},
        data={"consent": "true", "denoise": "true"},
    )
    assert response.status_code == 200
    assert "experimental Wiener denoising" in response.json()["audio"]["preprocessing"]


def test_comparison_and_robustness_report_explicitly_show_not_evaluated() -> None:
    assert request("GET", "/api/comparison").status_code == 200
    assert request("GET", "/api/robustness").status_code == 200
    assert request("GET", "/api/cross-corpus").status_code == 200
    assert request("GET", "/api/dataset-audit").status_code == 200


def test_model_card_and_capabilities_are_available_without_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(service, "model", None)
    monkeypatch.setattr(service, "threshold", None)
    monkeypatch.setattr(service, "architecture", None)
    card = request("GET", "/api/model-card").json()
    capabilities = request("GET", "/api/capabilities").json()
    assert card["status"] == "pretrained_available"
    assert card["pretrained_model"]["class_labels"] == ["neutral", "happy", "angry", "sad"]
    assert card["pretrained_model"]["download_on_first_use"] is True
    assert capabilities["local_audio_features"] is True
    assert capabilities["emotion_prediction"] is True
    assert capabilities["pretrained_audio_embeddings"] is True
    assert capabilities["cross_corpus"] is False
    assert capabilities["raw_audio_retained"] is False


def test_analysis_endpoint_accepts_pretrained_model_selection(monkeypatch) -> None:
    wav = BytesIO()
    sf.write(wav, [0.2] * 16_000, 16_000, format="WAV")
    monkeypatch.setattr(
        service,
        "analyze",
        lambda audio, **kwargs: {"selected_architecture": kwargs["architecture"]},
    )
    response = request(
        "POST",
        "/api/analyze",
        files={"file": ("recording.wav", wav.getvalue())},
        data={"consent": "true", "architecture": PRETRAINED_ARCHITECTURE},
    )
    assert response.status_code == 200
    assert response.json()["selected_architecture"] == PRETRAINED_ARCHITECTURE


def test_unsupported_extension_is_rejected() -> None:
    response = request(
        "POST",
        "/api/analyze",
        files={"file": ("recording.mp3", b"audio")},
        data={"consent": "true"},
    )
    assert response.status_code == 415


def test_silent_wav_upload_returns_clear_validation_error() -> None:
    wav = BytesIO()
    sf.write(wav, [0.0] * 16_000, 16_000, format="WAV")
    response = request(
        "POST",
        "/api/analyze",
        files={"file": ("silence.wav", wav.getvalue())},
        data={"consent": "true"},
    )
    assert response.status_code in (422, 503)
    if response.status_code == 422:
        assert "silent" in response.json()["detail"].lower()


def test_valid_wav_returns_features_without_claiming_emotion_prediction(monkeypatch) -> None:
    monkeypatch.setattr(service, "model", None)
    monkeypatch.setattr(service, "threshold", None)
    monkeypatch.setattr(service, "architecture", None)
    wav = BytesIO()
    sf.write(wav, [0.1] * 16_000, 16_000, format="WAV")
    response = request(
        "POST",
        "/api/analyze",
        files={"file": ("valid.wav", wav.getvalue())},
        data={"consent": "true"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "features_only"
    assert result["summary"] is None
    assert result["features"]["log_mel"]


def test_crema_d_named_audio_does_not_get_ravdess_prediction() -> None:
    wav = BytesIO()
    sf.write(wav, [0.1] * 16_000, 16_000, format="WAV")
    response = request(
        "POST",
        "/api/analyze",
        files={"file": ("1001_IEO_ANG_HI.wav", wav.getvalue())},
        data={"consent": "true"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "features_only"
    assert result["analysis_mode"] == "unsupported_corpus_features_only"
    assert result["source_corpus_hint"] == "CREMA-D"
    assert result["summary"] is None
    assert result["filename_annotation"]["label"] == "angry"
    assert "not an EmotionLens model prediction" in result["filename_annotation"]["note"]
    assert "CREMA-D model" in result["analysis_warning"]


def test_analysis_requires_explicit_consent() -> None:
    response = request(
        "POST",
        "/api/analyze",
        files={"file": ("recording.wav", b"audio")},
        data={"consent": "false"},
    )
    assert response.status_code == 400
