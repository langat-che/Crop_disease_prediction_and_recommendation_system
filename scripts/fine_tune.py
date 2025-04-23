# Scripts/fine_tune.py
# (Using the complete code block from the previous message, starting with '# Scripts/fine_tune.py' and ending with 'fine_tune_model(args)')
import os
import argparse
import tensorflow as tf
import pickle
import json
import pathlib
import time

# Import utility functions from utils.py
try:
    import utils
except ImportError:
    print("Error: Could not import utils.py. Ensure it's in the same directory or Python path.")
    print("If running from project root, try: python Scripts/fine_tune.py ...")
    exit(1)


def fine_tune_model(args):
    """Handles the fine-tuning process."""
    print(f"--- Starting Fine-Tuning for Model: {args.initial_model_path} ---")
    start_time = time.time()

    # --- 1. Load Initial Model ---
    initial_model_path = pathlib.Path(args.initial_model_path)
    if not initial_model_path.is_file():
        print(f"ERROR: Initial model not found at {initial_model_path}")
        return False

    print(f"\nStep 1: Loading initial model from {initial_model_path}...")
    try:
        model = tf.keras.models.load_model(initial_model_path)
        print("  Model loaded successfully.")
        # Extract info needed later
        try:
             input_shape = model.input_shape[1:] # (height, width, channels)
             num_classes_loaded = model.output_shape[-1]
             img_height_loaded, img_width_loaded, _ = input_shape
             print(f"  Detected Input Shape: ({img_height_loaded}, {img_width_loaded})")
             print(f"  Detected Number of Classes: {num_classes_loaded}")
             # Override args with detected values if defaults were used
             if args.img_height == utils.IMG_HEIGHT_DEFAULT: args.img_height = img_height_loaded
             if args.img_width == utils.IMG_WIDTH_DEFAULT: args.img_width = img_width_loaded
        except Exception as e:
             print(f"  Warning: Could not reliably determine input shape or classes from loaded model: {e}")
             num_classes_loaded = None # Reset if detection failed

    except Exception as e:
        print(f"ERROR loading initial model: {e}")
        return False

    # --- 2. Get Paths and Classes ---
    print("\nStep 2: Getting paths and class information...")
    train_dir, test_dir, class_names, num_classes = utils.get_paths_and_classes(args.data_base_dir, args.crop_type)
    if num_classes == 0:
        print("ERROR: Could not find classes or directories. Exiting.")
        return False
    if num_classes_loaded is not None and num_classes_loaded != num_classes:
        print(f"ERROR: Number of classes from loaded model ({num_classes_loaded}) differs from data directory ({num_classes}). Fine-tuning cannot proceed.")
        return False
    if args.img_height != img_height_loaded or args.img_width != img_width_loaded:
         print(f"ERROR: Specified image size ({args.img_height}x{args.img_width}) differs from loaded model input ({img_height_loaded}x{img_width_loaded}). Adjust arguments to match model.")
         return False
    img_size = (args.img_height, args.img_width) # Use verified/consistent size

    print(f"Using Image Size: {img_size}")
    print(f"Using Class Names: {class_names}")

    # --- 3. Load Datasets ---
    print("\nStep 3: Loading and configuring datasets...")
    apply_augmentation = args.augmentation_strength.lower() != 'none'
    print(f"Augmentation Enabled: {apply_augmentation} (Strength: {args.augmentation_strength})")

    train_ds = utils.configure_dataset(
        train_dir, img_size, args.batch_size, class_names,
        augment=apply_augmentation,
        augmentation_strength=args.augmentation_strength,
        shuffle_files=True
    )
    val_ds = utils.configure_dataset(
        test_dir, img_size, args.batch_size, class_names,
        augment=False, shuffle_files=False
    )
    if train_ds is None or val_ds is None:
        print("ERROR: Failed to load datasets. Exiting.")
        return False

    # --- 4. Partially Unfreeze Base Model ---
    print("\nStep 4: Unfreezing base model layers for fine-tuning...")
    try:
        # Assumes standard structure: Input -> Functional Base Model -> Head
        if len(model.layers) > 1 and isinstance(model.layers[1], tf.keras.Model):
             base_model = model.layers[1]
        else: # Fallback if structure is different (e.g. Sequential wrapper)
             # This requires knowing the base model's name or finding it
             base_model = model.get_layer(index=1) # Or find by name if known
             if not base_model or not hasattr(base_model, 'layers'):
                  raise ValueError("Could not identify base model layer automatically.")

        base_model.trainable = True # Unfreeze all first
        num_base_layers = len(base_model.layers)
        print(f"  Total layers in base model ('{base_model.name}'): {num_base_layers}")

        fine_tune_at = args.fine_tune_at
        if fine_tune_at < 0 or fine_tune_at >= num_base_layers:
            print(f"  fine_tune_at ({fine_tune_at}) is out of range [0, {num_base_layers-1}]. Unfreezing ALL base layers.")
            fine_tune_at = 0 # Unfreeze all

        if fine_tune_at > 0:
            print(f"  Freezing layers 0 to {fine_tune_at - 1}...")
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            print("  Layers frozen.")
        else:
             print("  All base model layers will be trainable.")

    except Exception as e:
        print(f"ERROR accessing/modifying base model layers: {e}")
        return False

    # --- 5. Re-compile with Low Learning Rate ---
    print("\nStep 5: Re-compiling model with low learning rate...")
    optimizer_fine = tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr)
    model.compile(optimizer=optimizer_fine,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    print(f"  Compiled with Adam, LR={args.fine_tune_lr}")
    model.summary(print_fn=print)

    # --- 6. Define Callbacks ---
    print("\nStep 6: Defining fine-tuning callbacks...")
    model_save_dir = pathlib.Path(args.model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    # Create a descriptive filename for the fine-tuned model
    ft_checkpoint_filename = (
        f"{args.crop_type}_finetuned_L{args.fine_tune_at}" # Indicate unfreeze point
        f"_ftlr{args.fine_tune_lr}"      # Fine-tune LR (NEW - Standard Float)
        f"_from_{initial_model_path.stem}" # Indicate source model
        f".keras"
    )
    ft_checkpoint_path = model_save_dir / ft_checkpoint_filename

    ft_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ft_checkpoint_path,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks_list = [ft_checkpoint_callback]
    print(f"  ModelCheckpoint: Monitoring 'val_accuracy', saving best to '{ft_checkpoint_path}'")

    if args.early_stopping_patience > 0:
        ft_early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            verbose=1,
            restore_best_weights=True
        )
        callbacks_list.append(ft_early_stopping_callback)
        print(f"  EarlyStopping: Monitoring 'val_loss', patience={args.early_stopping_patience}, restoring best weights.")

    # --- 7. Continue Training (Fine-tuning) ---
    print(f"\nStep 7: Starting fine-tuning for {args.fine_tune_epochs} epochs...")
    try:
        # We don't load previous history, so epochs count from 0 for this run
        history_fine = model.fit(
            train_ds,
            epochs=args.fine_tune_epochs, # Run only for the specified fine-tune epochs
            validation_data=val_ds,
            callbacks=callbacks_list,
            verbose=args.verbose
        )
        print("\n--- Fine-Tuning Finished ---")
        # Note: 'Best' model based on val_accuracy was saved by checkpoint.
        # The 'model' variable might hold weights restored by EarlyStopping (based on val_loss).
        print(f"Best fine-tuned model (by val_accuracy) saved to {ft_checkpoint_path}")

    except Exception as e:
        print(f"\nERROR during fine-tuning: {e}")
        return False

    # --- 8. Save/Plot History (Optional) ---
    if args.history_save_path:
        hist_path = pathlib.Path(args.history_save_path)
        hist_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        hist_filename = ft_checkpoint_path.stem + "_history.json"
        hist_save_full_path = hist_path / hist_filename
        print(f"\nStep 8: Saving fine-tuning history to {hist_save_full_path}...")
        try:
            history_dict = {key: [float(v) for v in values] for key, values in history_fine.history.items()}
            with open(hist_save_full_path, 'w') as f:
                json.dump(history_dict, f)
            print("  History saved successfully.")
            # Plotting
            plot_filename = ft_checkpoint_path.stem + "_history.png"
            plot_save_path = hist_path / plot_filename
            print(f"  Generating history plot (saving to {plot_save_path})...")
            # Plot only fine-tuning history, maybe add prefix to title
            utils.plot_history(history_fine, save_path=plot_save_path, title_prefix=f"{args.crop_type} Fine-Tune")

        except Exception as e:
             print(f"  Error saving/plotting history: {e}")

    end_time = time.time()
    print(f"\nTotal script duration: {(end_time - start_time) / 60:.2f} minutes.")
    return True

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained initial model for crop classification.")
    # Required Args
    parser.add_argument("--initial_model_path", type=str, required=True, help="Path to the initially trained '.keras' model file.")
    parser.add_argument("--crop_type", type=str, required=True, choices=['maize', 'onion', 'tomato'], help="Type of crop (must match the initial model).")
    parser.add_argument("--data_base_dir", type=str, required=True, help="Path to the main 'Data' directory.")
    parser.add_argument("--fine_tune_epochs", type=int, required=True, help="Number of epochs for fine-tuning.")
    parser.add_argument("--fine_tune_lr", type=float, required=True, help="Learning rate for fine-tuning (e.g., 1e-5).")
    parser.add_argument("--fine_tune_at", type=int, required=True, help="Layer index in base model from which to start unfreezing (e.g., 100). Use 0 or negative to unfreeze all.")
    parser.add_argument("--model_save_dir", type=str, required=True, help="Directory to save the best fine-tuned model.")
    # Optional Args
    parser.add_argument("--img_height", type=int, default=utils.IMG_HEIGHT_DEFAULT, help=f"Image height (default: {utils.IMG_HEIGHT_DEFAULT}). Used if detection fails or override needed.")
    parser.add_argument("--img_width", type=int, default=utils.IMG_WIDTH_DEFAULT, help=f"Image width (default: {utils.IMG_WIDTH_DEFAULT}). Used if detection fails or override needed.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--augmentation_strength", type=str, default='mild', choices=['mild', 'strong', 'geometric', 'none'], help="Level of data augmentation (default: 'mild').")
    parser.add_argument("--history_save_path", type=str, default=None, help="Optional: Directory to save fine-tuning history JSON and plot PNG.")
    parser.add_argument("--early_stopping_patience", type=int, default=7, help="Patience for EarlyStopping based on val_loss. 0 disables (default: 7).")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Verbosity mode for model.fit (default: 1).")

    args = parser.parse_args()
    fine_tune_model(args)