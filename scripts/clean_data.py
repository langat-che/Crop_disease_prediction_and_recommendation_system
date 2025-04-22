# Scripts/clean_data.py
import os
import argparse
import pathlib
import time

# Import utility functions from utils.py
try:
    import utils
except ImportError:
    print("Error: Could not import utils.py. Ensure it's in the same directory or Python path.")
    print("If running from project root, try: python Scripts/clean_data.py ...")
    exit(1)

# --- Configuration ---
# Define problematic characters and their replacements
FILENAME_REPLACEMENTS = {
    'é': 'e', 'É': 'E', 'ó': 'o', 'Ó': 'O',
    'à': 'a', 'À': 'A', 'ç': 'c', 'Ç': 'c',
    'ü': 'u', 'Ü': 'U', 'ñ': 'n', 'Ñ': 'N',
    # Add others if found necessary based on dataset characters
}


# --- Helper Functions ---
def check_and_fix_filenames(target_dir_str, fix=False):
    """Scans and optionally renames files/dirs with non-ASCII chars."""
    target_dir = pathlib.Path(target_dir_str)
    print(f"\nScanning for non-ASCII names in: {target_dir}")
    fixed_count = 0
    problematic_found = 0
    skipped_exist = 0
    errors = 0

    items_to_check = list(target_dir.rglob('*')) # Get all files and directories recursively

    for item_path in items_to_check:
        original_name = item_path.name
        new_name = original_name
        needs_rename = False

        for char_from, char_to in FILENAME_REPLACEMENTS.items():
            if char_from in new_name:
                new_name = new_name.replace(char_from, char_to)
                needs_rename = True

        if needs_rename:
            problematic_found += 1
            new_path = item_path.parent / new_name
            print(f"  Problematic name found: {item_path}")
            if fix:
                if item_path != new_path:
                    try:
                        if new_path.exists():
                            print(f"    SKIPPING rename: Target '{new_path.name}' already exists.")
                            skipped_exist += 1
                        else:
                            item_path.rename(new_path)
                            print(f"    RENAMED to: '{new_path.name}'")
                            fixed_count += 1
                    except OSError as e:
                        print(f"    ERROR renaming {item_path.name}: {e}")
                        errors += 1
                else:
                     pass # No actual change needed
            else:
                print("    (Fixing disabled, rename manually or use --fix_filenames)")

    print(f"Filename Scan Summary for {target_dir}:")
    print(f"  Problematic names found: {problematic_found}")
    if fix:
        print(f"  Successfully renamed: {fixed_count}")
        print(f"  Skipped (target existed): {skipped_exist}")
        print(f"  Errors during rename: {errors}")
    return errors == 0

def check_image_corruption(target_dir_str, delete_corrupted=False):
    """Uses utils.is_image_file_valid to check and optionally delete bad files."""
    target_dir = pathlib.Path(target_dir_str)
    print(f"\nScanning for corrupted images in: {target_dir}")
    corrupted_count = 0
    deleted_count = 0
    checked_count = 0
    errors = 0

    for item_path in target_dir.rglob('*'):
        # Check if it's a file and has a supported image extension
        if item_path.is_file() and item_path.suffix.lower() in utils.SUPPORTED_IMAGE_EXTENSIONS:
            checked_count += 1
            if not utils.is_image_file_valid(item_path):
                corrupted_count += 1
                print(f"  Corrupted image detected: {item_path}")
                if delete_corrupted:
                    try:
                        item_path.unlink() # Delete the file
                        print("    DELETED.")
                        deleted_count += 1
                    except OSError as e:
                        print(f"    ERROR deleting {item_path}: {e}")
                        errors += 1
                else:
                     print("    (Deletion disabled, delete manually or use --delete_corrupted)")

    print(f"Corruption Check Summary for {target_dir}:")
    print(f"  Images checked: {checked_count}")
    print(f"  Corrupted detected: {corrupted_count}")
    if delete_corrupted:
        print(f"  Successfully deleted: {deleted_count}")
        print(f"  Errors during deletion: {errors}")
    return errors == 0

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean image dataset directories by checking filenames and image integrity.")
    parser.add_argument("--data_base_dir", type=str, required=True, help="Path to the main 'Data' directory containing crop subdirectories.")
    parser.add_argument("--crop_type", type=str, required=True, help="Crop type ('maize', 'onion', 'tomato', or 'all').")
    parser.add_argument("--fix_filenames", action='store_true', help="Automatically rename files/directories with non-ASCII characters based on FILENAME_REPLACEMENTS.")
    parser.add_argument("--skip_corruption_check", action='store_true', help="Skip the check for corrupted image files.")
    parser.add_argument("--delete_corrupted", action='store_true', help="Delete corrupted image files if found (only works if corruption check is enabled). USE WITH CAUTION!")

    args = parser.parse_args()
    start_time_main = time.time()

    base_dir = pathlib.Path(args.data_base_dir)
    if not base_dir.is_dir():
        print(f"ERROR: Data base directory not found: {args.data_base_dir}")
        exit(1)

    crops_to_process = []
    if args.crop_type.lower() == 'all':
        crops_to_process = ['maize', 'onion', 'tomato']
    elif args.crop_type.lower() in ['maize', 'onion', 'tomato']:
        crops_to_process = [args.crop_type.lower()]
    else:
        print(f"ERROR: Invalid crop_type '{args.crop_type}'.")
        exit(1)

    overall_success = True
    for crop in crops_to_process:
        print(f"\n--- Processing Crop: {crop.upper()} ---")
        train_dir, test_dir, _, _ = utils.get_paths_and_classes(args.data_base_dir, crop)

        if train_dir:
            print(f"\n>>> Processing TRAIN directory: {train_dir}")
            if not check_and_fix_filenames(str(train_dir), args.fix_filenames): overall_success = False
            if not args.skip_corruption_check:
                 if not check_image_corruption(str(train_dir), args.delete_corrupted): overall_success = False
        else:
             print(f"WARNING: Skipping train directory processing for {crop} (path not found or invalid).")

        if test_dir:
            print(f"\n>>> Processing TEST directory: {test_dir}")
            if not check_and_fix_filenames(str(test_dir), args.fix_filenames): overall_success = False
            if not args.skip_corruption_check:
                 if not check_image_corruption(str(test_dir), args.delete_corrupted): overall_success = False
        else:
            print(f"WARNING: Skipping test directory processing for {crop} (path not found or invalid).")

    end_time_main = time.time()
    print("\n--- Cleaning Process Finished ---")
    if not overall_success:
        print("WARNING: Errors occurred during the cleaning process. Please review the logs above.")
    else:
        print("Cleaning process completed.")
    print(f"Total duration: {(end_time_main - start_time_main) / 60:.2f} minutes.")