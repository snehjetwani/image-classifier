"""
FastAPI REST API for the image classifier.

Run with:  uvicorn api:app --reload
Docs:      http://localhost:8000/docs
"""
import os
import tempfile

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from classifier import load_model, predict as clf_predict

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_model = None
_class_names: list[str] = []
_model_path_cache: str | None = None

MODEL_PATH: str = os.environ.get("MODEL_PATH", "model.pth")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Image Classifier API",
    description="REST API for classifying images with a trained ResNet18 model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PredictionResult(BaseModel):
    predicted: str
    confidence: float
    scores: dict[str, float]


class BatchItem(BaseModel):
    filename: str
    predicted: str | None = None
    confidence: float | None = None
    scores: dict[str, float] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------
def get_model():
    """Ensure the model is loaded; raise 503 if the file is missing."""
    global _model, _class_names, _model_path_cache

    if _model is not None and MODEL_PATH == _model_path_cache:
        return _model, _class_names

    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — check MODEL_PATH",
        )

    _model, _class_names = load_model(MODEL_PATH)
    _model._loaded_from = MODEL_PATH
    _model_path_cache = MODEL_PATH
    return _model, _class_names


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Return service health and basic model info."""
    num_classes = len(_class_names) if _class_names else 0
    return {"status": "ok", "model": MODEL_PATH, "classes": num_classes}


@app.get("/model/info")
def model_info(model_and_names=Depends(get_model)):
    """Return architecture details and parameter counts for the loaded model."""
    model, class_names = model_and_names
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    file_size_mb = (
        round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 4)
        if os.path.isfile(MODEL_PATH)
        else None
    )
    return {
        "architecture": "ResNet18",
        "num_classes": len(class_names),
        "classes": class_names,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "file_size_mb": file_size_mb,
    }


@app.post("/predict", response_model=PredictionResult)
async def predict_image(
    file: UploadFile = File(...),
    model_and_names=Depends(get_model),
):
    """Classify a single uploaded image and return the top prediction with scores."""
    model, class_names = model_and_names

    suffix = os.path.splitext(file.filename or "image.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        result = clf_predict(model, class_names, tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Inference error: {exc}") from exc
    finally:
        os.remove(tmp_path)

    predicted = result["predicted"]
    confidence = result["scores"][predicted]
    return PredictionResult(
        predicted=predicted,
        confidence=round(confidence, 4),
        scores={cls: round(score, 4) for cls, score in result["scores"].items()},
    )


@app.post("/batch", response_model=list[BatchItem])
async def predict_batch(
    files: list[UploadFile] = File(...),
    model_and_names=Depends(get_model),
):
    """Classify multiple uploaded images; errors are captured per file rather than raised."""
    model, class_names = model_and_names
    results: list[BatchItem] = []

    for upload in files:
        filename = upload.filename or "unknown"
        suffix = os.path.splitext(filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(await upload.read())

        try:
            result = clf_predict(model, class_names, tmp_path)
            predicted = result["predicted"]
            confidence = result["scores"][predicted]
            results.append(
                BatchItem(
                    filename=filename,
                    predicted=predicted,
                    confidence=round(confidence, 4),
                    scores={cls: round(score, 4) for cls, score in result["scores"].items()},
                )
            )
        except Exception as exc:
            results.append(BatchItem(filename=filename, error=str(exc)))
        finally:
            os.remove(tmp_path)

    return results
