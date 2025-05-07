# Scripts/evaluate.py
import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pathlib
import time

# Import utility functions from utils.py
try:
    import utils
except ImportError:
    print("Error: Could not import utils.py. Ensure it's in the same directory or Python path.")
    print("If running from project root, try: python Scripts/evaluate.py ...")
    exit(1)

def evaluate_saved_model(args):
    """Loads a saved model and evaluates it on the test set."""
    print(f"--- Starting Evaluation for Model: {args.model_path} ---")
    start_time = time.time()

    model_path = pathlib.Path(args.model_path)
    if not model_path.is_file():
        print(f"ERROR: Model file not found at {model_path}")
        return False

    # --- 1. Get Paths and Classes ---
    print("\nStep 1: Getting paths and class information...")
    # We need the test_dir and class_names
    _, test_dir, class_names, num_classes = utils.get_paths_and_classes(args.data_base_dir, args.crop_type)
    if num_classes == 0 or class_names is None:
        print("ERROR: Could not find classes or test directory. Exiting.")
        return False

    img_size = (args.img_height, args.img_width)
    print(f"Using Image Size: {img_size}")
    print(f"Using Class Names: {class_names}")

    # --- 2. Load Test Dataset ---
    print("\nStep 2: Loading and configuring test dataset...")
    # IMPORTANT: No shuffling, no augmentation for evaluation set
    test_ds = utils.configure_dataset(
        test_dir, img_size, args.batch_size, class_names,
        augment=False,
        shuffle_files=False # Ensure order is consistent for confusion matrix
    )
    if test_ds is None:
        print("ERROR: Failed to load test dataset. Exiting.")
        return False

    # --- 3. Load the Model ---
    print(f"\nStep 3: Loading model from {model_path}...")
    try:
        # Specify custom objects if needed (though usually not for standard layers)
        # model = tf.keras.models.load_model(model_path, custom_objects={...})
        model = tf.keras.models.load_model(model_path)
        print("  Model loaded successfully.")
        # Optional: Verify input/output shapes match expectations
        try:
            model_input_shape = model.input_shape[1:]
            model_output_classes = model.output_shape[-1]
            if model_input_shape[:2] != img_size:
                 print(f"  WARNING: Model input HxW {model_input_shape[:2]} differs from specified img_size {img_size}. Evaluation might be incorrect if resizing didn't happen correctly.")
            if model_output_classes != num_classes:
                 print(f"  WARNING: Model output classes ({model_output_classes}) differs from dataset classes ({num_classes}). Report labels might be wrong.")
        except Exception as e:
             print(f"  Warning: Could not verify model input/output shapes: {e}")

    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

    # --- 4. Evaluate Model (Loss & Accuracy) ---
    print("\nStep 4: Evaluating model performance...")
    try:
        eval_results = model.evaluate(test_ds, verbose=args.verbose, return_dict=True)
        print("\n--- Overall Performance ---")
        print(f"  Test Loss: {eval_results['loss']:.4f}")
        print(f"  Test Accuracy: {eval_results['accuracy']:.4f} ({(eval_results['accuracy']*100):.2f}%)")
    except Exception as e:
         print(f"ERROR during model evaluation: {e}")
         # Continue to try generating reports if possible
         eval_results = {'loss': -1, 'accuracy': -1} # Placeholder

    # --- 5. Generate Predictions for Reports ---
    print("\nStep 5: Generating predictions for detailed reports...")
    y_pred_probs = []
    y_true_labels = []
    try:
        # Iterate through the test dataset to get all predictions and labels
        item_count = 0
        for images, labels in test_ds:
            batch_preds = model.predict_on_batch(images)
            y_pred_probs.extend(batch_preds)
            y_true_labels.extend(labels.numpy())
            item_count += len(labels.numpy())
        print(f"  Generated predictions for {item_count} samples.")

        # Convert probabilities and one-hot labels to class indices
        y_pred_indices = np.argmax(y_pred_probs, axis=1)
        y_true_indices = np.argmax(y_true_labels, axis=1)

        # Verify using sklearn accuracy (should match TF evaluate)
        sk_accuracy = accuracy_score(y_true_indices, y_pred_indices)
        print(f"  Scikit-learn Accuracy (Verification): {sk_accuracy:.4f}")
        if eval_results['accuracy'] > 0 and not np.isclose(eval_results['accuracy'], sk_accuracy, atol=1e-4):
             print(f"  WARNING: TF Evaluate accuracy ({eval_results['accuracy']:.4f}) and sklearn accuracy ({sk_accuracy:.4f}) differ slightly.")

    except Exception as e:
        print(f"ERROR during prediction generation: {e}")
        # Cannot generate reports if predictions fail
        return False

    # --- 6. Generate Classification Report ---
    print("\nStep 6: Generating Classification Report...")
    report_save_full_path = None
    if args.report_save_path:
        report_path = pathlib.Path(args.report_save_path)
        report_path.mkdir(parents=True, exist_ok=True)
        report_filename = model_path.stem + "_classification_report.txt"
        report_save_full_path = report_path / report_filename

    try:
        report = classification_report(y_true_indices, y_pred_indices, target_names=class_names, digits=4)
        print(report)
        # Save report if path provided
        if report_save_full_path:
            try:
                with open(report_save_full_path, 'w') as f:
                    f.write(f"Classification Report for Model: {model_path.name}\n")
                    f.write(f"Evaluated on: {test_dir}\n")
                    f.write(f"Overall Loss: {eval_results['loss']:.4f}\n")
                    f.write(f"Overall Accuracy: {eval_results['accuracy']:.4f}\n\n")
                    f.write(report)
                print(f"  Classification report saved to: {report_save_full_path}")
            except Exception as e_save:
                print(f"  Error saving classification report: {e_save}")
    except Exception as e:
        print(f"ERROR generating classification report: {e}")

    # --- 7. Generate Confusion Matrix ---
    print("\nStep 7: Generating Confusion Matrix...")
    cm_save_full_path = None
    if args.cm_save_path:
        cm_path = pathlib.Path(args.cm_save_path)
        cm_path.mkdir(parents=True, exist_ok=True)
        cm_filename = model_path.stem + "_confusion_matrix.png"
        cm_save_full_path = cm_path / cm_filename

    try:
        utils.plot_confusion_matrix_util(
            y_true_indices, y_pred_indices, class_names,
            save_path=cm_save_full_path,
            title_prefix=f"{args.crop_type} ({model_path.stem})"
            )
    except Exception as e:
        print(f"ERROR generating confusion matrix: {e}")

    end_time = time.time()
    print(f"\nTotal evaluation duration: {(end_time - start_time) / 60:.2f} minutes.")
    return True

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained crop classification model.")
    # Required Args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved '.keras' model file to evaluate.")
    parser.add_argument("--crop_type", type=str, required=True, choices=['maize', 'onion', 'tomato'], help="Crop type (for finding test data and class names).")
    parser.add_argument("--data_base_dir", type=str, required=True, help="Path to the main 'Data' directory.")
    # Optional Args
    parser.add_argument("--img_height", type=int, default=utils.IMG_HEIGHT_DEFAULT, help=f"Image height (default: {utils.IMG_HEIGHT_DEFAULT}). Must match model input.")
    parser.add_argument("--img_width", type=int, default=utils.IMG_WIDTH_DEFAULT, help=f"Image width (default: {utils.IMG_WIDTH_DEFAULT}). Must match model input.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation (doesn't affect result, only speed/memory). Default: 32.")
    parser.add_argument("--report_save_path", type=str, default=None, help="Optional: Directory to save the classification report text file (e.g., '../reports').")
    parser.add_argument("--cm_save_path", type=str, default=None, help="Optional: Directory to save the confusion matrix plot PNG file (e.g., '../reports').")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1], help="Verbosity for model.evaluate (0 = silent, 1 = progress bar). Default: 1.")

    args = parser.parse_args()
    evaluate_saved_model(args)
