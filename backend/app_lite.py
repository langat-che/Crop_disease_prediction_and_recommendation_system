import os
import sys
import time
import logging
import gc

# ── Force CPU-only TensorFlow (Render free tier has no GPU) ──
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# ── Logging Setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ── Configuration ──
IMG_SIZE = (160, 160)
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best_models"))
EXPECTED_MODELS = {
    "maize":  "maize_model.keras",
    "onion":  "onion_model.keras",
    "tomato": "tomato_model.keras"
}
CLASS_NAMES = {
    "maize":  ['Abiotics_diseases_d','Aphids_p','Healthy_leaf','Rust_d','Spodoptera_frugiperda_a',
               'Spodoptera_frugiperda_p','curvulariosis_d','helminthosporiosis_d','stripe_d'],
    "onion":  ['Bulb_blight_d','Healthy_leaf','alternaria_d','caterpillars_p','fusarium_d','virosis_d'],
    "tomato": ['Healthy_fruit','Mite_P','Tomato_late_blight_d','alternaria_d','alternaria_mite_d',
               'bacterial_floundering_d','blossom_end_rot_d','fusarium_d','healthy_leaf',
               'helicoverpa_armigera_p','nitrogen_exces_d','sunburn_d','tuta_absoluta_p','virosis_d']
}

# ── Verify & Preload Models at Startup ──
models = {}
def verify_and_load_models():
    logger.info(f"Verifying model directory exists at: {MODEL_DIR}")
    if not os.path.isdir(MODEL_DIR):
        logger.critical(f"Model directory not found: {MODEL_DIR}")
        sys.exit(1)

    files = os.listdir(MODEL_DIR)
    logger.info(f"Found files in model dir: {files}")

    for crop, fname in EXPECTED_MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.isfile(path):
            logger.critical(f"Missing model file for '{crop}': {fname}")
            sys.exit(1)
        try:
            logger.info(f"Loading '{crop}' model from {fname}...")
            models[crop] = tf.keras.models.load_model(path)
            logger.info(f"✅ '{crop}' model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load '{fname}' for '{crop}': {e}")
            sys.exit(1)

    logger.info("✅✅✅ All models verified and loaded into memory.")

verify_and_load_models()

# ── FastAPI App ──
app = FastAPI(title="Crop Disease Predictor API (Lite)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # consider locking this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "loaded_models": list(models.keys())}

@app.get("/")
async def root():
    return {
        "message": "Crop Disease Predictor API (Lite) is running.",
        "available_models": list(models.keys())
    }

@app.post("/predict")
async def predict_disease(
    crop: str = Form(...),
    file: UploadFile = File(...)
):
    start_time = time.time()
    crop_key = crop.lower()
    logger.info(f"Received request – crop: '{crop_key}', file: '{file.filename}'")

    if crop_key not in models:
        logger.warning(f"Invalid or unloaded crop: '{crop_key}'")
        raise HTTPException(400, detail=f"Invalid crop '{crop}'. Available: {list(models.keys())}")

    # Preprocess image
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = img.resize(IMG_SIZE, Image.Resampling.NEAREST)
        arr = np.array(img, dtype=np.float32) / 255.0
        input_tensor = np.expand_dims(arr, 0)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(400, detail="Invalid image data.")

    # Predict
    model = models[crop_key]
    try:
        preds = model.predict(input_tensor, verbose=0)[0]
    except Exception as e:
        logger.error(f"Model prediction error: {e}")
        raise HTTPException(500, detail="Prediction failed.")

    try:
        idx = int(np.argmax(preds))
        label = CLASS_NAMES[crop_key][idx]
        confidence = float(preds[idx])
    except Exception as e:
        logger.error(f"Postprocessing error: {e}")
        raise HTTPException(500, detail="Failed to interpret prediction.")

    duration = time.time() - start_time
    logger.info(f"Prediction for '{crop_key}': {label} ({confidence:.4f}) in {duration:.3f}s")

    # LRU-style eviction if memory pressure
    if len(models) > 2:
        logger.info(f"Clearing '{crop_key}' from cache to save memory.")
        del models[crop_key]
        gc.collect()

    return {"crop": crop_key, "prediction": label, "confidence": round(confidence,4)}

# ── Uvicorn Entry ──
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
