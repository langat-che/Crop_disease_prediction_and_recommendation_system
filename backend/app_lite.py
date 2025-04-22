# backend/app_lite.py - Lightweight version for Render free tier
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import logging
import sys
import time

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(title="Crop Disease Predictor API (Lite)")

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Try multiple possible model directories to be more robust
possible_model_dirs = [
    os.path.join(project_root, 'best_models'),  # Standard local path
    '/opt/render/project/src/best_models',      # Render.com path with disk mount
    os.path.join(project_root, 'models'),       # Fallback to models directory
    '/app/best_models',                         # Another common path in containerized environments
    os.path.abspath('best_models')              # Absolute path as last resort
]

# Find the first directory that exists
MODEL_DIR = None
for dir_path in possible_model_dirs:
    logger.info(f"Checking for models in: {dir_path}")
    if os.path.exists(dir_path):
        MODEL_DIR = dir_path
        logger.info(f"Found models directory at: {MODEL_DIR}")
        # List files in the directory for debugging
        try:
            files = os.listdir(dir_path)
            logger.info(f"Files in {dir_path}: {files}")
        except Exception as e:
            logger.error(f"Error listing files in {dir_path}: {e}")
        break

if MODEL_DIR is None:
    logger.critical("Could not find any valid models directory!")
    # Default to first option even if it doesn't exist
    MODEL_DIR = possible_model_dirs[0]
    logger.info(f"Using default model directory: {MODEL_DIR}")

# Image size MUST match the training size
IMG_SIZE = (160, 160)

# --- Load Models ---
# We'll load models on-demand to save memory
models = {}
expected_models = {
    "maize": "maize_model.keras",
    "onion": "onion_model.keras",
    "tomato": "tomato_model.keras"
}

# --- Class Names ---
CLASS_NAMES = {
    "maize": ['Abiotics_diseases_d', 'Aphids_p', 'Healthy_leaf', 'Rust_d', 'Spodoptera_frugiperda_a', 'Spodoptera_frugiperda_p', 'curvulariosis_d', 'helminthosporiosis_d', 'stripe_d'],
    "onion": ['Bulb_blight_d', 'Healthy_leaf', 'alternaria_d', 'caterpillars_p', 'fusarium_d', 'virosis_d'],
    "tomato": ['Healthy_fruit', 'Mite_P', 'Tomato_late_blight_d', 'alternaria_d', 'alternaria_mite_d', 'bacterial_floundering_d', 'blossom_end_rot_d', 'fusarium_d', 'healthy_leaf', 'helicoverpa_armigera_p', 'nitrogen_exces_d', 'sunburn_d', 'tuta_absoluta_p', 'virosis_d']
}

# --- Helper Functions ---
def load_model(crop_key):
    """Load model on demand to save memory"""
    if crop_key in models:
        return models[crop_key]

    filename = expected_models.get(crop_key)
    if not filename:
        return None

    model_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None

    try:
        logger.info(f"Loading model for '{crop_key}' from {filename}...")
        model = tf.keras.models.load_model(model_path)
        models[crop_key] = model
        logger.info(f"Successfully loaded model for '{crop_key}'.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint for health check."""
    # Check which model files exist without loading them
    available_models = []
    for crop, filename in expected_models.items():
        if os.path.exists(os.path.join(MODEL_DIR, filename)):
            available_models.append(crop)

    return {
        "message": "Crop Disease Predictor API (Lite) is running.",
        "available_models": available_models,
        "loaded_models": list(models.keys())
    }

@app.post("/predict")
async def predict_disease(
    crop: str = Form(...),
    file: UploadFile = File(...)
):
    """Predicts disease for a given crop image."""
    start_time = time.time()
    logger.info(f"Received prediction request for crop: '{crop}', file: '{file.filename}'")

    crop_key = crop.lower()

    # --- Input Validation ---
    if crop_key not in expected_models:
        logger.warning(f"Invalid crop type received: '{crop}'")
        raise HTTPException(status_code=400, detail=f"Invalid crop type '{crop}'. Available: {list(expected_models.keys())}")

    if crop_key not in CLASS_NAMES:
        logger.error(f"Class names not configured for crop type '{crop}'")
        raise HTTPException(status_code=500, detail=f"Internal configuration error: Class names missing for '{crop}'.")

    # --- Load Model ---
    model = load_model(crop_key)
    if model is None:
        raise HTTPException(status_code=500, detail=f"Failed to load model for '{crop}'.")

    # --- Image Preprocessing ---
    try:
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from file.")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMG_SIZE, Image.Resampling.NEAREST)
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = np.expand_dims(arr, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    # --- Prediction ---
    try:
        preds = model.predict(tensor, verbose=0)[0]
        predicted_index = int(np.argmax(preds))
        confidence = float(preds[predicted_index])

        class_names_for_crop = CLASS_NAMES[crop_key]
        if predicted_index >= len(class_names_for_crop):
            logger.error(f"Predicted index {predicted_index} is out of bounds")
            predicted_label = "Prediction Index Error"
        else:
            predicted_label = class_names_for_crop[predicted_index]

        logger.info(f"Prediction: '{predicted_label}', Confidence: {confidence:.4f}")

        # Clear model from memory if we're running low on memory
        # Uncomment this if you're having memory issues
        # if len(models) > 1:  # Keep at least one model in memory
        #     del models[crop_key]
        #     import gc
        #     gc.collect()
        #     logger.info(f"Cleared model '{crop_key}' from memory to save resources")

        return {
            "crop": crop,
            "prediction": predicted_label,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Uvicorn startup ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
