# Scripts/utils.py
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pathlib
import os
import time
from PIL import Image # Added for robust image check

# --- Constants ---
IMG_HEIGHT_DEFAULT = 160
IMG_WIDTH_DEFAULT = 160
AUTOTUNE = tf.data.AUTOTUNE
SUPPORTED_IMAGE_EXTENSIONS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')

# --- Path and Class Handling ---
def get_paths_and_classes(data_base_dir_str, crop_type):
    """
    Constructs paths to train/test data and gets sorted class names.

    Args:
        data_base_dir_str (str): Path to the main 'Data' directory.
        crop_type (str): Name of the crop ('maize', 'onion', 'tomato').

    Returns:
        tuple: (train_dir_path, test_dir_path, class_names, num_classes)
               Returns (None, None, None, 0) if paths are invalid.
    """
    data_base_dir = pathlib.Path(data_base_dir_str)
    # Define explicit mapping for folder names
    crop_folder_map = {
        'maize': 'maize dataset with data augmentation',
        'onion': 'onion with data augmentation',
        'tomato': 'tomato_with_data_augmentation'
    }
    crop_folder_name = crop_folder_map.get(crop_type.lower())

    if not crop_folder_name:
        print(f"ERROR: Unknown crop_type '{crop_type}'. Use 'maize', 'onion', or 'tomato'.")
        return None, None, None, 0

    crop_dir = data_base_dir / crop_folder_name
    train_dir = crop_dir / 'train'
    test_dir = crop_dir / 'test' # Assuming validation/test split is named 'test'

    if not train_dir.is_dir():
        print(f"ERROR: Training directory not found for {crop_type} at: {train_dir.resolve()}")
        return None, None, None, 0
    if not test_dir.is_dir():
         print(f"ERROR: Test directory not found for {crop_type} at: {test_dir.resolve()}")
         # Allow returning train dir even if test is missing, maybe for train-only tasks
         # return train_dir, None, None, 0
         # For now, require both:
         return None, None, None, 0

    try:
        # List only directories as classes
        class_names = sorted([item.name for item in train_dir.iterdir() if item.is_dir()])
        num_classes = len(class_names)
        if num_classes == 0:
            print(f"WARNING: No class subdirectories found in {train_dir}")
            return train_dir, test_dir, [], 0
        print(f"Found {num_classes} classes for {crop_type}: {class_names}")
        # Verify test directory also contains (at least some of) the same classes if needed
        return train_dir, test_dir, class_names, num_classes
    except Exception as e:
        print(f"Error listing classes in {train_dir}: {e}")
        return train_dir, test_dir, None, 0


# --- Data Augmentation Definitions ---
def get_data_augmentation(strength='mild', img_height=IMG_HEIGHT_DEFAULT, img_width=IMG_WIDTH_DEFAULT):
    """Gets a Sequential model for data augmentation based on strength."""
    augmentation_layers = [
         layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3))
         # Ensure input_shape is only on the first layer if using Sequential this way
    ]
    name_suffix = ""

    if strength == 'mild':
        augmentation_layers.extend([
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        name_suffix = "mild"
    elif strength == 'strong':
        augmentation_layers.extend([
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.15), # Slightly adjusted factors
            layers.RandomBrightness(0.15)
        ])
        name_suffix = "strong"
    elif strength == 'geometric':
         augmentation_layers.extend([
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ])
         name_suffix = "geometric"
    elif strength == 'none':
         return None # Return None if no augmentation needed
    else:
         print(f"Warning: Unknown augmentation strength '{strength}'. Using 'mild'.")
         return get_data_augmentation('mild', img_height, img_width)

    # Build the Sequential model only if layers were added
    if len(augmentation_layers) > 1: # Needs at least flip + one more
         return tf.keras.Sequential(augmentation_layers, name=f"data_augmentation_{name_suffix}")
    else: # Case where only flip might be defined, handle appropriately or return None
         return None


# --- Dataset Configuration ---
def configure_dataset(dir_path, img_size, batch_size, class_names, augment=False, augmentation_strength='mild', shuffle_files=True, seed=42):
    """Loads, preprocesses, and configures a dataset with optimized pipeline."""
    if not dir_path or not class_names:
         print("ERROR: configure_dataset requires valid dir_path and class_names.")
         return None

    print(f"  Configuring dataset from: {dir_path}")
    print(f"  Image Size: {img_size}, Batch Size: {batch_size}, Shuffle Files: {shuffle_files}, Augment: {augment} ({augmentation_strength})")

    try:
        dataset = tf.keras.utils.image_dataset_from_directory(
            dir_path,
            labels='inferred',
            label_mode='categorical',
            class_names=class_names,
            image_size=img_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=shuffle_files,
            seed=seed
        )
        print(f"  Found {tf.data.experimental.cardinality(dataset)*batch_size if tf.data.experimental.cardinality(dataset) > 0 else 'unknown'} files belonging to {len(class_names)} classes.")
    except Exception as e:
        print(f"ERROR loading dataset from {dir_path}: {e}")
        return None

    # Define preprocessing functions
    rescale_layer = layers.Rescaling(1./255)
    def apply_rescaling(image, label):
        # Ensure image is float32 before rescaling
        image = tf.cast(image, tf.float32)
        return rescale_layer(image), label

    # Configure dataset pipeline
    ds = dataset.map(apply_rescaling, num_parallel_calls=AUTOTUNE)

    # Cache after rescaling
    ds = ds.cache()
    print("  Dataset caching enabled.")

    # Apply augmentation AFTER caching if enabled
    if augment and augmentation_strength != 'none':
        augmentation_layer = get_data_augmentation(augmentation_strength, img_size[0], img_size[1])
        if augmentation_layer:
            def apply_augmentation(image, label):
                return augmentation_layer(image, training=True), label
            ds = ds.map(apply_augmentation, num_parallel_calls=AUTOTUNE)
            print(f"  Applied '{augmentation_strength}' augmentation.")
        else:
            print(f"  Warning: Augmentation requested ('{augmentation_strength}') but no layers generated.")

    # Shuffle training data buffer AFTER caching/augmentation
    if shuffle_files: # Typically True for training
        buffer_size = tf.data.experimental.cardinality(ds) # Get number of batches
        if buffer_size > 0:
            # Use dataset size or a large number for buffer
            shuffle_buffer = max(buffer_size.numpy() * batch_size // 2 , 1000)
        else:
            shuffle_buffer = 1000 # Fallback buffer size
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        print(f"  Dataset shuffling enabled (buffer={shuffle_buffer}).")

    # Always prefetch for performance
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    print("  Dataset prefetching enabled.")

    return ds


# --- Model Building ---
def build_transfer_model(img_height, img_width, num_classes, dropout_rate=0.2, base_model_trainable=False, fine_tune_at=0):
    """Builds the MobileNetV2 transfer learning model."""
    print(f"\n--- Building Model ---")
    print(f"  Input Size: ({img_height}x{img_width}x3)")
    print(f"  Num Classes: {num_classes}")
    print(f"  Dropout Rate: {dropout_rate}")
    print(f"  Base Model Trainable: {base_model_trainable}")

    # Load base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )

    # Set trainability of base model
    base_model.trainable = base_model_trainable
    if base_model_trainable and fine_tune_at > 0:
        num_base_layers = len(base_model.layers)
        if fine_tune_at >= num_base_layers:
             print(f"  WARNING: fine_tune_at ({fine_tune_at}) >= num layers ({num_base_layers}). Unfreezing all base layers.")
        else:
            print(f"  Fine-tuning: Freezing base model layers before layer index {fine_tune_at}...")
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            print(f"  Layers 0 to {fine_tune_at-1} frozen.")
    elif not base_model_trainable:
        print("  Base model frozen.")
    else:
        print("  All base model layers are trainable.")


    # Define inputs and build the head
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    # Pass input through base model - use training=False if base is frozen or partially frozen
    # to keep batch norm layers in inference mode for those parts
    x = base_model(inputs, training=base_model_trainable) # Only set training=True if fully unfrozen for fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print("  Model constructed.")
    return model


# --- Plotting Functions ---
def plot_history(history_data, initial_epochs=None, save_path=None, title_prefix=""):
    """Plots training/validation accuracy and loss from history data (dict or History object)."""
    if isinstance(history_data, tf.keras.callbacks.History):
        history_dict = history_data.history
    elif isinstance(history_data, dict):
        history_dict = history_data
    else:
        print("Error: Invalid history_data type provided for plotting. Must be dict or History object.")
        return

    acc = history_dict.get('accuracy', [])
    val_acc = history_dict.get('val_accuracy', [])
    loss = history_dict.get('loss', [])
    val_loss = history_dict.get('val_loss', [])

    if not all([acc, val_acc, loss, val_loss]):
        print("Warning: History object missing required keys ('accuracy', 'val_accuracy', 'loss', 'val_loss'). Cannot plot.")
        return

    epochs_run = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))
    title_prefix = title_prefix + " " if title_prefix else ""

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_run, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs_run, val_acc, 'r-', label='Validation Accuracy')
    best_val_acc_epoch = np.argmax(val_acc) + 1
    best_val_acc = np.max(val_acc)
    plt.scatter(best_val_acc_epoch, best_val_acc, color='red', marker='*', s=100, label=f'Best Val Acc ({best_val_acc:.3f})')

    if initial_epochs and initial_epochs < len(acc):
        plt.axvline(initial_epochs, linestyle='--', color='gray', label=f'Start Fine-Tune')
    plt.title(f'{title_prefix}Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_run, loss, 'b-', label='Training Loss')
    plt.plot(epochs_run, val_loss, 'r-', label='Validation Loss')
    best_val_loss_epoch = np.argmin(val_loss) + 1
    best_val_loss = np.min(val_loss)
    plt.scatter(best_val_loss_epoch, best_val_loss, color='red', marker='*', s=100, label=f'Best Val Loss ({best_val_loss:.3f})')

    if initial_epochs and initial_epochs < len(acc):
        plt.axvline(initial_epochs, linestyle='--', color='gray', label=f'Start Fine-Tune')
    plt.title(f'{title_prefix}Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.suptitle(f'{title_prefix}Training and Validation Metrics ({len(acc)} Epochs)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        try:
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved history plot to: {save_path}")
        except Exception as e:
            print(f"Error saving history plot: {e}")
    plt.show()
    plt.close() # Close the plot to prevent display issues in loops


def plot_confusion_matrix_util(y_true_indices, y_pred_indices, class_names, save_path=None, title_prefix=""):
     """Calculates and plots a confusion matrix."""
     if len(y_true_indices) == 0 or len(y_pred_indices) == 0:
          print("ERROR: Cannot plot confusion matrix with empty true or predicted labels.")
          return
     title_prefix = title_prefix + " " if title_prefix else ""
     cm = confusion_matrix(y_true_indices, y_pred_indices)
     plt.figure(figsize=(max(6, len(class_names)*0.8), max(5, len(class_names)*0.6))) # Adjust size based on num classes
     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=class_names, yticklabels=class_names)
     plt.xlabel('Predicted Label')
     plt.ylabel('True Label')
     plt.title(f'{title_prefix}Confusion Matrix')
     plt.xticks(rotation=45, ha='right')
     plt.yticks(rotation=0)
     plt.tight_layout()

     if save_path:
        try:
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved confusion matrix plot to: {save_path}")
        except Exception as e:
            print(f"Error saving confusion matrix plot: {e}")
     plt.show()
     plt.close()


# --- Single Image Preprocessing ---
def preprocess_single_image(image_path, img_height=IMG_HEIGHT_DEFAULT, img_width=IMG_WIDTH_DEFAULT):
    """Loads and preprocesses a single image for prediction."""
    try:
        img_path_str = str(image_path) # Ensure it's a string for tf.io
        img = tf.io.read_file(img_path_str)
        # Decode any common format; expand_animations prevents issues with GIFs
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        # Resize using 'nearest' to match training preprocessing if that was used
        img = tf.image.resize(img, [img_height, img_width], method='nearest')
        # Cast to float32 BEFORE rescaling
        img = tf.cast(img, tf.float32) / 255.0  # Rescale to [0,1]
        img = tf.expand_dims(img, axis=0)  # Add batch dimension: (1, H, W, C)
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None


# --- Robust Image Check ---
def is_image_file_valid(filepath):
    """Checks if an image file can be opened and read by PIL."""
    try:
        img = Image.open(filepath)
        img.verify()  # Verify verifies headers, decodes enough to check validity
        img.close() # Close file handle
        # Additionally try reading a pixel (more thorough but slower)
        # img = Image.open(filepath)
        # img.load()
        # img.close()
        return True
    except (IOError, SyntaxError, Exception) as e:
        #print(f"  Invalid image file {filepath}: {e}") # Optional verbose logging
        return False