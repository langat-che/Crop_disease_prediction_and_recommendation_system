# Scripts/predict.py
import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time

# Import utility functions from utils.py
try:
    import utils
except ImportError:
    print("Error: Could not import utils.py. Ensure it's in the same directory or Python path.")
    print("If running from project root, try: python Scripts/predict.py ...")
    exit(1)

def predict_single_image(args):
    """Loads a model and predicts the class for a single image."""
    print(f"--- Predicting Image: {args.image_path} ---")
    start_time = time.time()

    model_path = pathlib.Path(args.model_path)
    image_path = pathlib.Path(args.image_path)

    if not model_path.is_file():
        print(f"ERROR: Model file not found at {model_path}")
        return False
    if not image_path.is_file():
        print(f"ERROR: Image file not found at {image_path}")
        return False

    # --- 1. Load the Model ---
    print(f"\nStep 1: Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("  Model loaded successfully.")
        # Detect input size from model if args aren't overriding
        try:
            input_shape = model.input_shape[1:] # (height, width, channels)
            # Use args if they are not the default, otherwise use detected shape
            img_height_load = args.img_height if args.img_height != utils.IMG_HEIGHT_DEFAULT else input_shape[0]
            img_width_load = args.img_width if args.img_width != utils.IMG_WIDTH_DEFAULT else input_shape[1]
            img_size_load = (img_height_load, img_width_load)
            print(f"  Using image size: ({img_size_load[0]} x {img_size_load[1]})")

        except Exception as e:
             print(f"  Warning: Could not determine model input size, using specified args ({args.img_height}x{args.img_width}). Error: {e}")
             img_size_load = (args.img_height, args.img_width)

    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

    # --- 2. Load and Preprocess Image ---
    print(f"\nStep 2: Loading and preprocessing image {image_path.name}...")
    img_array = utils.preprocess_single_image(str(image_path), img_size_load[0], img_size_load[1])

    if img_array is None:
        print("ERROR: Failed to preprocess image.")
        return False
    print(f"  Image preprocessed to shape: {img_array.shape}")

    # --- 3. Make Prediction ---
    print("\nStep 3: Making prediction...")
    try:
        predictions = model.predict(img_array, verbose=0)
        pred_probs = predictions[0]
    except Exception as e:
         print(f"ERROR during model prediction: {e}")
         return False

    # --- 4. Interpret Prediction ---
    print("\nStep 4: Interpreting results...")
    # --- Get Class Names ---
    class_names = None
    num_model_outputs = len(pred_probs)

    if args.class_names:
        # Prioritize explicitly provided names
        class_names = [name.strip() for name in args.class_names.split(',')]
        print(f"  Using provided class names list (count: {len(class_names)})")
        if len(class_names) != num_model_outputs:
             print(f"  WARNING: Number of provided class names ({len(class_names)}) does not match model output size ({num_model_outputs}). Ensure correct names/order are provided.")
             # Adjust list length to match model output if mismatched
             if len(class_names) > num_model_outputs: class_names = class_names[:num_model_outputs]
             else: class_names.extend([f"Unknown_{i}" for i in range(num_model_outputs - len(class_names))])

    elif args.crop_type and args.data_base_dir:
         # Fallback: try to get class names from the data directory
         print(f"  Attempting to get class names for crop '{args.crop_type}' from data directory...")
         _, _, class_names_from_data, num_classes_from_data = utils.get_paths_and_classes(args.data_base_dir, args.crop_type)
         if class_names_from_data and len(class_names_from_data) == num_model_outputs:
              class_names = class_names_from_data
              print(f"  Using class names from data directory: {class_names}")
         else:
             print(f"  WARNING: Could not retrieve matching class names from data dir (found {len(class_names_from_data) if class_names_from_data else 0}, expected {num_model_outputs}). Using generic names.")
             class_names = [f"Class_{i}" for i in range(num_model_outputs)]
    else:
        # Final fallback: generic names
        print("  WARNING: No class names provided or derivable. Using generic names.")
        class_names = [f"Class_{i}" for i in range(num_model_outputs)]

    # Find the winning class index and its probability
    predicted_index = np.argmax(pred_probs)
    predicted_prob = np.max(pred_probs)
    try:
        predicted_class_name = class_names[predicted_index]
    except IndexError:
         predicted_class_name = f"Index_{predicted_index}_(Name Error)"


    print("\n--- Prediction Result ---")
    print(f"  Predicted Class: {predicted_class_name}")
    print(f"  Confidence: {predicted_prob:.4f} ({(predicted_prob*100):.2f}%)")

    # Show image and top predictions if requested
    if args.show_plot or args.top_n > 1:
        # Print top N predictions
        top_n = min(args.top_n, len(class_names))
        # Get indices of top N probabilities, sorted highest to lowest
        top_indices = np.argsort(pred_probs)[-top_n:][::-1]
        print(f"\n  Top {top_n} Predictions:")
        for i in top_indices:
             try:
                 print(f"    - {class_names[i]}: {pred_probs[i]:.4f}")
             except IndexError:
                  print(f"    - Index_{i}_(Name Error): {pred_probs[i]:.4f}")

    if args.show_plot:
        try:
            # Display the image - load again in displayable format
            img_display = tf.io.read_file(str(image_path))
            img_display = tf.io.decode_image(img_display, channels=3, expand_animations=False)
            plt.figure(figsize=(6, 6))
            plt.imshow(img_display.numpy())
            plt.title(f"Predicted: {predicted_class_name} ({predicted_prob:.2f})")
            plt.axis("off")
            plt.show()
        except Exception as e_plot:
            print(f"Error displaying plot: {e_plot}")


    end_time = time.time()
    print(f"\nTotal prediction duration: {end_time - start_time:.2f} seconds.")
    return True

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict class for a single image using a trained crop classification model.")
    # Required Args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved '.keras' model file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the single image file to predict.")
    # Class Names Handling - User must provide one method
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--class_names", type=str, help="Comma-separated string of class names IN THE ORDER THE MODEL EXPECTS THEM (e.g., 'ClassA,ClassB,ClassC').")
    group.add_argument("--crop_type", type=str, choices=['maize', 'onion', 'tomato'], help="Crop type, used WITH --data_base_dir to automatically get class names from the training data structure.")
    parser.add_argument("--data_base_dir", type=str, help="Path to the main 'Data' directory (REQUIRED if using --crop_type instead of --class_names).")
    # Optional Args
    parser.add_argument("--img_height", type=int, default=utils.IMG_HEIGHT_DEFAULT, help=f"Image height (default: {utils.IMG_HEIGHT_DEFAULT}). If different from default, overrides model detection.")
    parser.add_argument("--img_width", type=int, default=utils.IMG_WIDTH_DEFAULT, help=f"Image width (default: {utils.IMG_WIDTH_DEFAULT}). If different from default, overrides model detection.")
    parser.add_argument("--show_plot", action='store_true', help="Display the input image with the prediction title.")
    parser.add_argument("--top_n", type=int, default=3, help="Show top N predictions (default: 3). Automatically enabled if --show_plot is used.")

    args = parser.parse_args()

    # Validate arguments for class names
    if args.crop_type and not args.data_base_dir:
        parser.error("--data_base_dir is required when using --crop_type to derive class names.")
    if args.show_plot and args.top_n < 1 : # Ensure top_n is at least 1 if plotting
         args.top_n = 1

    predict_single_image(args)
