#!/usr/bin/env python
# setup_models.py - Script to copy models to the correct location during deployment

import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_models():
    """Copy models from source to destination directories if needed."""
    # Define possible source and destination directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    source_dirs = [
        os.path.join(current_dir, 'best_models'),
        os.path.join(current_dir, 'models')
    ]
    
    dest_dirs = [
        os.path.join(current_dir, 'best_models'),
        '/opt/render/project/src/best_models'
    ]
    
    # Find first valid source directory with model files
    source_dir = None
    for dir_path in source_dirs:
        if os.path.exists(dir_path) and any(f.endswith('.keras') for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))):
            source_dir = dir_path
            logger.info(f"Found source models in: {source_dir}")
            break
    
    if not source_dir:
        logger.error("No source directory with model files found!")
        return False
    
    # Copy to all destination directories that don't already have the models
    success = False
    for dest_dir in dest_dirs:
        if dest_dir == source_dir:
            logger.info(f"Skipping copy to {dest_dir} (same as source)")
            success = True
            continue
            
        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)
        
        # Check if models already exist in destination
        if any(f.endswith('.keras') for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))):
            logger.info(f"Models already exist in {dest_dir}, skipping copy")
            success = True
            continue
        
        # Copy all .keras files
        try:
            for filename in os.listdir(source_dir):
                if filename.endswith('.keras'):
                    src_file = os.path.join(source_dir, filename)
                    dst_file = os.path.join(dest_dir, filename)
                    logger.info(f"Copying {src_file} to {dst_file}")
                    shutil.copy2(src_file, dst_file)
            logger.info(f"Successfully copied models to {dest_dir}")
            success = True
        except Exception as e:
            logger.error(f"Error copying models to {dest_dir}: {e}")
    
    return success

if __name__ == "__main__":
    logger.info("Starting model setup process...")
    if setup_models():
        logger.info("Model setup completed successfully")
    else:
        logger.error("Model setup failed")
