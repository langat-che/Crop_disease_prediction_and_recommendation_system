# Scripts/train.py
import os
import argparse
import tensorflow as tf
import pickle
import json
import pathlib
import time

# Import utility functions from utils.py (assuming it's in the same directory)
try:
    import utils
except ImportError:
    print("Error: Could not import utils.py. Ensure it's in the same directory or Python path.")
    print("If running from project root, try: python Scripts/train.py ...")
    exit(1)

def train_initial_model(args):
    """
    Handles the initial training process (frozen base model).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    print(f"--- Starting Initial Training for Crop: {args.crop_type.upper()} ---")
    start_time = time.time()

    # --- 1. Get Paths and Classes ---
    print("\nStep 1: Getting paths and class information...")
    train_dir, test_dir, class_names, num_classes = utils.get_paths_and_classes(args.data_base_dir, args.crop_type)
    if num_classes == 0:
        print("ERROR: Could not find classes or directories. Exiting.")
        return False # Indicate failure

    img_size = (args.img_height, args.img_width)
    print(f"Image Size: {img_size}")
    print(f"Batch Size: {args.batch_size}")

    # --- 2. Load and Configure Datasets ---
    print("\nStep 2: Loading and configuring datasets...")
    # Determine if augmentation should be applied
    apply_augmentation = args.augmentation_strength.lower() != 'none'
    print(f"Augmentation Enabled: {apply_augmentation} (Strength: {args.augmentation_strength})")

    train_ds = utils.configure_dataset(
        train_dir, img_size, args.batch_size, class_names,
        augment=apply_augmentation, # Pass boolean based on strength arg
        augmentation_strength=args.augmentation_strength,
        shuffle_files=True
    )
    val_ds = utils.configure_dataset(
        test_dir, img_size, args.batch_size, class_names,
        augment=False, # Never augment validation/test set
        shuffle_files=False # No shuffling for validation/test set
    )
    if train_ds is None or val_ds is None:
        print("ERROR: Failed to load datasets. Exiting.")
        return False

    # --- 3. Build Model ---
    print("\nStep 3: Building model with frozen base...")
    # Use base_model_trainable=False for initial training
    model = utils.build_transfer_model(
        args.img_height, args.img_width, num_classes, args.dropout_rate,
        base_model_trainable=False
        )

    # --- 4. Compile Model ---
    print("\nStep 4: Compiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary(print_fn=print) # Print summary to console

    # --- 5. Define Callbacks ---
    print("\nStep 5: Defining callbacks...")
    # Ensure model save directory exists
    model_save_dir = pathlib.Path(args.model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    # Define checkpoint path based on parameters (more descriptive)
    # Inside train_initial_model function in train.py
    checkpoint_filename = (
        f"{args.crop_type}_initial_best"
        f"_img{args.img_height}"
        f"_dr{args.dropout_rate:.1f}" # Keep dropout format
        # f"_lr{args.learning_rate:.0e}" # OLD - Remove scientific notation
        f"_lr{args.learning_rate}"      # NEW - Use standard float representation
        f"_aug_{args.augmentation_strength}"
        f".keras"
    )
    checkpoint_path = model_save_dir / checkpoint_filename

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False, # Save the full model
        monitor='val_accuracy',  # Monitor validation accuracy
        mode='max',              # Save the model with maximum val_accuracy
        save_best_only=True,     # Only save if it's the best so far
        verbose=1
    )
    callbacks_list = [model_checkpoint_callback]
    print(f"  ModelCheckpoint: Monitoring 'val_accuracy', saving best to '{checkpoint_path}'")

    # Optional Early Stopping
    if args.early_stopping_patience > 0:
        # Restore best weights based on val_loss for early stopping
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            verbose=1,
            restore_best_weights=True
        )
        callbacks_list.append(early_stopping_callback)
        print(f"  EarlyStopping: Monitoring 'val_loss', patience={args.early_stopping_patience}, restoring best weights.")

    # --- 6. Train Model ---
    print(f"\nStep 6: Starting training for {args.epochs} epochs...")
    try:
        history = model.fit(
            train_ds,
            epochs=args.epochs,
            validation_data=val_ds,
            callbacks=callbacks_list,
            verbose=args.verbose # Control training verbosity
        )
        print("\n--- Training Finished ---")
        # Note: 'Best' model based on val_accuracy was saved by checkpoint during training
        # The 'model' variable might hold weights restored by EarlyStopping (based on val_loss) if it triggered.
        print(f"Best model (by val_accuracy) saved to {checkpoint_path}")

    except Exception as e:
        print(f"\nERROR during model training: {e}")
        # Potentially save history even if training failed mid-way
        if 'history' in locals() and args.history_save_path:
             # Try saving partial history
             # ... (add saving logic here if needed) ...
             pass
        return False

    # --- 7. Save History (Optional) ---
    if args.history_save_path:
        # Use same base name as model checkpoint for consistency
        hist_path = pathlib.Path(args.history_save_path)
        hist_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        hist_filename = checkpoint_path.stem + "_history.json"
        hist_save_full_path = hist_path / hist_filename

        print(f"\nStep 7: Saving training history to {hist_save_full_path}...")
        try:
            # Save as JSON (more portable than pickle)
            # Convert numpy types to standard types for JSON serialization
            history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
            with open(hist_save_full_path, 'w') as f:
                json.dump(history_dict, f)
            print("  History saved successfully.")
        except Exception as e:
             print(f"  Error saving history: {e}")

    # --- 8. Plot History (Optional) ---
    plot_save_path = None
    if args.history_save_path: # Only attempt plot save if history was saved
        plot_filename = checkpoint_path.stem + "_history.png"
        plot_save_path = hist_path / plot_filename
        print(f"\nStep 8: Generating history plot (saving to {plot_save_path})...")
    else:
        print("\nStep 8: Generating history plot...")

    utils.plot_history(history, save_path=plot_save_path, title_prefix=f"{args.crop_type} Initial")

    end_time = time.time()
    print(f"\nTotal script duration: {(end_time - start_time) / 60:.2f} minutes.")
    return True # Indicate success

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train initial model (frozen base) for crop classification.")
    # Required Args
    parser.add_argument("--crop_type", type=str, required=True, choices=['maize', 'onion', 'tomato'], help="Type of crop to train.")
    parser.add_argument("--data_base_dir", type=str, required=True, help="Path to the main 'Data' directory containing crop subdirectories (e.g., '../Data').")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for initial training.")
    parser.add_argument("--dropout_rate", type=float, required=True, help="Dropout rate for the classification head (e.g., 0.4).")
    parser.add_argument("--model_save_dir", type=str, required=True, help="Directory to save the best trained model (e.g., '../models').")
    # Optional Args with Defaults
    parser.add_argument("--img_height", type=int, default=utils.IMG_HEIGHT_DEFAULT, help=f"Image height (default: {utils.IMG_HEIGHT_DEFAULT}).")
    parser.add_argument("--img_width", type=int, default=utils.IMG_WIDTH_DEFAULT, help=f"Image width (default: {utils.IMG_WIDTH_DEFAULT}).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for Adam optimizer (default: 0.001).")
    parser.add_argument("--augmentation_strength", type=str, default='mild', choices=['mild', 'strong', 'geometric', 'none'], help="Level of data augmentation (default: 'mild').")
    parser.add_argument("--history_save_path", type=str, default=None, help="Optional: Directory to save training history JSON and plot PNG (e.g., '../training_history').")
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Patience for EarlyStopping based on val_loss. 0 disables early stopping (default: 0).")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Verbosity mode for model.fit (0 = silent, 1 = progress bar, 2 = one line per epoch). Default: 1.")

    args = parser.parse_args()
    train_initial_model(args)
