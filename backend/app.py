# backend/app.py
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
import time  # Added missing import for time module

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(title="Crop Disease Predictor API")

# --- CORS Configuration ---
# Allows frontend hosted elsewhere (like Netlify, Vercel, GitHub Pages) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, restrict in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
# Determine paths relative to the current file's location (backend/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Go up one level to project root

# Try multiple possible model directories to be more robust in different environments
possible_model_dirs = [
    os.path.join(project_root, 'best_models'),  # Standard local path
    '/opt/render/project/src/best_models',      # Render.com path with disk mount
    os.path.join(project_root, 'models')        # Fallback to models directory
]

# Find the first directory that exists
MODEL_DIR = None
for dir_path in possible_model_dirs:
    if os.path.exists(dir_path):
        MODEL_DIR = dir_path
        logger.info(f"Found models directory at: {MODEL_DIR}")
        break

if MODEL_DIR is None:
    logger.critical("Could not find any valid models directory!")
    MODEL_DIR = possible_model_dirs[0]  # Default to first option even if it doesn't exist

# Image size MUST match the training size
IMG_SIZE = (160, 160)

# --- Load Models ---
models = {}
expected_models = {
    "maize": "maize_model.keras",
    "onion": "onion_model.keras",
    "tomato": "tomato_model.keras"
}

logger.info(f"Attempting to load models from: {os.path.abspath(MODEL_DIR)}")
for crop, filename in expected_models.items():
    model_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(model_path):
        try:
            logger.info(f"Loading model for '{crop}' from {filename}...")
            models[crop] = tf.keras.models.load_model(model_path)
            logger.info(f"Successfully loaded model for '{crop}'.")
        except Exception as e:
            logger.error(f"ERROR: Failed to load model for '{crop}' from {model_path}: {e}", exc_info=True)
    else:
        logger.warning(f"WARNING: Model file not found for crop '{crop}' at {model_path}")

if not models:
    logger.critical("FATAL: No models were loaded successfully. API may not function correctly.")
    # sys.exit("Model loading failed, exiting.") # Optionally exit if models are critical

# --- Class Names (CRITICAL: UPDATE THESE MANUALLY) ---
# Replace these lists with the EXACT class names in the EXACT order your models were trained on.
# This order usually corresponds to the alphabetical order of the subdirectories in your training data.
CLASS_NAMES = {
    "maize": ['Abiotics_diseases_d', 'Aphids_p', 'Healthy_leaf', 'Rust_d', 'Spodoptera_frugiperda_a', 'Spodoptera_frugiperda_p', 'curvulariosis_d', 'helminthosporiosis_d', 'stripe_d'], # Verified from previous logs
    "onion": ['Bulb_blight_d', 'Healthy_leaf', 'alternaria_d', 'caterpillars_p', 'fusarium_d', 'virosis_d'], # Verified from previous logs
    "tomato": ['Healthy_fruit', 'Mite_P', 'Tomato_late_blight_d', 'alternaria_d', 'alternaria_mite_d', 'bacterial_floundering_d', 'blossom_end_rot_d', 'fusarium_d', 'healthy_leaf', 'helicoverpa_armigera_p', 'nitrogen_exces_d', 'sunburn_d', 'tuta_absoluta_p', 'virosis_d'] # Verified from previous logs
}
logger.info("Class names configured.")


# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Crop Disease Predictor API is running.", "loaded_models": list(models.keys())}

@app.post("/predict")
async def predict_disease(
    crop: str = Form(...),           # Get crop type from form data
    file: UploadFile = File(...)     # Get image file from form data
):
    """Predicts disease for a given crop image."""
    start_time = time.time()
    logger.info(f"Received prediction request for crop: '{crop}', file: '{file.filename}'")

    crop_key = crop.lower() # Use lowercase for dictionary keys

    # --- Input Validation ---
    if crop_key not in models:
        logger.warning(f"Invalid crop type received: '{crop}'")
        raise HTTPException(status_code=400, detail=f"Invalid crop type '{crop}'. Available: {list(models.keys())}")
    if crop_key not in CLASS_NAMES:
         logger.error(f"Class names not configured for crop type '{crop}'")
         raise HTTPException(status_code=500, detail=f"Internal configuration error: Class names missing for '{crop}'.")

    # --- Image Preprocessing ---
    try:
        contents = await file.read()
        logger.info(f"  Read {len(contents)} bytes from file.")
        image = Image.open(io.BytesIO(contents)).convert("RGB") # Convert ensures 3 channels
        image = image.resize(IMG_SIZE, Image.Resampling.NEAREST) # Resize (use NEAREST like TF default)
        arr = np.array(image, dtype=np.float32) / 255.0 # Convert to float32 numpy array and rescale
        tensor = np.expand_dims(arr, axis=0) # Add batch dimension -> (1, 160, 160, 3)
        logger.info(f"  Image preprocessed to shape: {tensor.shape}")
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded image. Please ensure it's a valid image file. Error: {e}")

    # --- Prediction ---
    try:
        model_to_use = models[crop_key]
        class_names_for_crop = CLASS_NAMES[crop_key]

        logger.info(f"  Running prediction using model for '{crop_key}'...")
        preds = model_to_use.predict(tensor, verbose=0)[0] # verbose=0 for predict
        logger.debug(f"  Raw predictions (probabilities): {preds}")

        predicted_index = int(np.argmax(preds))
        confidence = float(preds[predicted_index])

        # Ensure index is valid for class names list
        if predicted_index >= len(class_names_for_crop):
             logger.error(f"Predicted index {predicted_index} is out of bounds for '{crop_key}' class names (len {len(class_names_for_crop)}). Check model output shape and CLASS_NAMES.")
             predicted_label = "Prediction Index Error" # Fallback label
        else:
            predicted_label = class_names_for_crop[predicted_index]

        logger.info(f"  Prediction Result: '{predicted_label}' (Index: {predicted_index}), Confidence: {confidence:.4f}")

        end_time = time.time()
        logger.info(f"Prediction request completed in {end_time - start_time:.4f} seconds.")

        # Return results
        return {
            "crop": crop, # Return original case as received
            "prediction": predicted_label,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
         logger.error(f"Error during model prediction: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal server error.")

# --- Uvicorn startup for local development and Render deployment ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
