from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audio import AudioInputError, decode_wav
from .inference import InferenceService, PRETRAINED_ARCHITECTURE


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
BASE_DIR = Path(__file__).resolve().parent
service = InferenceService()
app = FastAPI(
    title="EmotionLens",
    description="Local, research-oriented voice emotion analysis.",
    version="1.0.0",
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_available": service.available,
        "pretrained_model_available": service.pretrained_available,
        "pretrained_model_loaded": service.pretrained_model is not None,
    }


@app.get("/api/model")
def model_info() -> dict[str, object]:
    return service.model_info()


@app.get("/api/evaluation")
def evaluation() -> dict[str, object]:
    return service.evaluation_results()


@app.get("/api/comparison")
def comparison() -> dict[str, object]:
    return service.artifact_report("comparison")


@app.get("/api/robustness")
def robustness() -> dict[str, object]:
    return service.artifact_report("robustness")


@app.get("/api/cross-corpus")
def cross_corpus() -> dict[str, object]:
    return service.artifact_report("cross_corpus")


@app.get("/api/dataset-audit")
def dataset_audit() -> dict[str, object]:
    return service.artifact_report("dataset_audit")


@app.get("/api/model-card")
def model_card() -> dict[str, object]:
    info = service.model_info()
    return {
        "status": (
            "available"
            if info["available"]
            else "pretrained_available"
            if service.pretrained_available
            else "unavailable"
        ),
        "model_version": info.get("model_version"),
        "architecture": info.get("architecture"),
        "dataset": info.get("dataset"),
        "pretrained_model": info["pretrained_model"],
        "limitations": info.get(
            "limitations",
            [
                "No trained model is installed.",
                "No evaluation metrics are currently available.",
            ],
        ),
        "intended_use": "Research and educational exploration of acted or consented speech.",
        "prohibited_use": "Do not use for diagnosis, hiring, discipline, surveillance, or other high-stakes decisions.",
    }


@app.get("/api/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "local_audio_features": True,
        "emotion_prediction": service.available or PRETRAINED_ARCHITECTURE in service.available_architectures(),
        "pretrained_emotion_classifier": service.pretrained_available,
        "energy_activity_estimate": True,
        "speaker_reference": True,
        "json_export": True,
        "csv_export": True,
        "printable_report": True,
        "trained_evaluation": service.evaluation_results().get("status") == "evaluated",
        "model_comparison": service.artifact_report("comparison").get("status") == "evaluated",
        "noise_robustness": service.artifact_report("robustness").get("status") == "evaluated",
        "dataset_audit": service.artifact_report("dataset_audit").get("status") == "audited",
        "crema_d_model": service.dataset == "CREMA-D",
        "cross_corpus": service.artifact_report("cross_corpus").get("status") == "evaluated",
        "multilingual": False,
        "multimodal": False,
        "pretrained_audio_embeddings": service.pretrained_available,
        "speaker_adaptation_evaluated": False,
        "raw_audio_retained": False,
    }


@app.post("/api/features")
async def analyze_features(
    file: UploadFile = File(...),
    consent: bool = Form(...),
    denoise: bool = Form(False),
) -> dict[str, object]:
    try:
        if Path(file.filename or "").suffix.lower() != ".wav":
            raise HTTPException(status_code=415, detail="Only WAV files are supported.")
        if not consent:
            raise HTTPException(
                status_code=400,
                detail="Explicit permission to analyse this recording is required.",
            )
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio files must be 10 MB or smaller.")
        audio = decode_wav(contents)
        return service.analyze_features(audio, denoise=denoise)
    except AudioInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    consent: bool = Form(...),
    include_attribution: bool = Form(False),
    architecture: str | None = Form(None),
    denoise: bool = Form(False),
) -> dict[str, object]:
    try:
        filename = file.filename or ""
        if Path(filename).suffix.lower() != ".wav":
            raise HTTPException(status_code=415, detail="Only WAV files are supported.")
        if not consent:
            raise HTTPException(
                status_code=400,
                detail="Explicit permission to analyse this recording is required.",
            )
        if architecture is not None and architecture not in service.available_architectures():
            raise HTTPException(
                status_code=422,
                detail="The selected model architecture is not available. Run training with --compare to install the comparison models.",
            )
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio files must be 10 MB or smaller.")
        decoded = decode_wav(contents)
        return service.analyze(
            decoded,
            include_attribution=include_attribution,
            architecture=architecture,
            input_filename=filename,
            denoise=denoise,
        )
    except AudioInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
