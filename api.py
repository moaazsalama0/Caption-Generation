"""
Run:
    uvicorn api:app --reload
"""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from model import CaptionModel

CHECKPOINT_PATH = "checkpoints/best_caption_model.pth"

app = FastAPI(title="Image Captioning API")
model = CaptionModel(CHECKPOINT_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    return {"caption": model.predict(image)}
