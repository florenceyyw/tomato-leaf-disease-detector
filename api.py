"""REST API for the tomato leaf disease classifier (advanced feature).

Exposes the trained model as HTTP endpoints using FastAPI, so the classifier can
be called programmatically by other software, not only through the Streamlit
interface. Run with:  uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /            health check and metadata
  GET  /classes     the list of disease classes the model can predict
  POST /predict     upload an image, receive the predicted class, confidence,
                    and the full probability distribution as JSON
"""
import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Tomato Leaf Disease Detection API",
    description="Classify tomato leaf diseases from an uploaded image.",
    version="1.0.0",
)

# Load the model and metadata once at start-up.
with open("deploy_meta.json") as f:
    META = json.load(f)
MODEL = tf.keras.models.load_model(META["model_file"])
CLASS_NAMES = META["class_names"]
IMG_SIZE = META["img_size"]


def _prepare(pil_img):
    """Resize to the model input size on the raw 0..255 scale (model preprocesses internally)."""
    img = pil_img.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    return np.array(img).astype("float32")


@app.get("/")
def root():
    """Health check and basic metadata."""
    return {
        "status": "ok",
        "model": META.get("model_name", "unknown"),
        "num_classes": len(CLASS_NAMES),
        "input_size": IMG_SIZE,
        "usage": "POST an image file to /predict",
    }


@app.get("/classes")
def classes():
    """Return the list of classes the model can predict."""
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an uploaded image and return the prediction as JSON."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    arr = _prepare(pil_img)
    probs = MODEL.predict(arr[np.newaxis, ...], verbose=0)[0]
    top_idx = int(np.argmax(probs))

    # full distribution, sorted high to low
    distribution = sorted(
        ({"class": CLASS_NAMES[i], "probability": round(float(probs[i]), 4)}
         for i in range(len(CLASS_NAMES))),
        key=lambda d: d["probability"], reverse=True,
    )
    return JSONResponse({
        "prediction": CLASS_NAMES[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "distribution": distribution,
    })
