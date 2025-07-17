#!/usr/bin/env python3
"""
Script to create an unsplit version of the merged_dataset.
Combines train and valid folders into a single all folder with unified annotations.
"""

import shutil
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_unsplit_dataset():
    """Create an unsplit version of the merged dataset."""
    
    # Define source dataset directory
    # The merged dataset lives in data/merged_dataset with separate
    # `train` and `valid` folders.  We want to create a unified
    # `all` folder *inside the same merged_dataset directory* so that
    # downstream code can reference a single location.

    source_dir = Path("data/merged_dataset")

    # Destination folder that will hold the unified images
    target_all_dir = source_dir / "all"
    target_all_dir.mkdir(exist_ok=True)

    # We keep all generated artefacts (annotations, summary) inside the
    # same merged_dataset directory for simplicity.
    target_dir = source_dir
    
    logger.info(f"Creating unsplit dataset in {target_dir}")
    
    # Expected annotation files
    train_csv = source_dir / "train_annotations.csv"
    valid_csv = source_dir / "valid_annotations.csv"

    if not train_csv.exists() or not valid_csv.exists():
        raise FileNotFoundError(
            f"Expected annotation files not found.\n"
            f"  Train CSV: {train_csv} (exists={train_csv.exists()})\n"
            f"  Valid CSV: {valid_csv} (exists={valid_csv.exists()})"
        )
    
    logger.info("Reading annotation files...")
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    
    # Update image paths to point to the new `all/` folder
    train_df['image_path'] = train_df['image_path'].str.replace('train/', 'all/', regex=False)
    valid_df['image_path'] = valid_df['image_path'].str.replace('valid/', 'all/', regex=False)
    
    # Combine annotations
    combined_df = pd.concat([train_df, valid_df], ignore_index=True)
    
    # Sort by image path for consistency
    combined_df = combined_df.sort_values('image_path').reset_index(drop=True)
    
    logger.info(f"Combined dataset: {len(combined_df)} images")
    
    # Copy images from the train directory
    train_dir = source_dir / "train"
    copied_count = 0
    
    if train_dir.exists():
        logger.info("Copying images from train directory...")
        for img_file in train_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                target_path = target_all_dir / img_file.name
                if not target_path.exists():
                    shutil.copy2(img_file, target_path)
                    copied_count += 1
        logger.info(f"Copied {copied_count} images from train directory")
    
    # Copy images from the valid directory
    valid_dir = source_dir / "valid"
    valid_copied_count = 0
    
    if valid_dir.exists():
        logger.info("Copying images from valid directory...")
        for img_file in valid_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                target_path = target_all_dir / img_file.name
                if not target_path.exists():
                    shutil.copy2(img_file, target_path)
                    valid_copied_count += 1
        logger.info(f"Copied {valid_copied_count} images from valid directory")
    
    total_copied = copied_count + valid_copied_count
    logger.info(f"Total images copied: {total_copied}")
    
    # Save combined annotations
    output_csv = target_dir / "all_annotations.csv"
    combined_df.to_csv(output_csv, index=False)
    logger.info(f"Saved combined annotations to {output_csv}")
    
    # Create summary file
    summary_content = f"""License Plate Dataset Unsplit Summary
==================================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Original dataset structure:
  - Training images: {len(train_df)}
  - Validation images: {len(valid_df)}

Combined unsplit dataset:
  - Total images: {len(combined_df)}
  - Images copied: {total_copied}

Plate text length distribution:
  - Min length: {combined_df['plate_text'].str.len().min()}
  - Max length: {combined_df['plate_text'].str.len().max()}
  - Average length: {combined_df['plate_text'].str.len().mean():.1f}

Character frequency (top 20):
"""
    
    # Calculate character frequencies
    all_chars = ''.join(combined_df['plate_text'])
    char_counts = {}
    for char in all_chars:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Sort by frequency
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    
    for char, count in sorted_chars[:20]:
        summary_content += f"  - '{char}': {count}\n"
    
    # Save summary
    summary_file = target_dir / "dataset_summary_all.txt"
    with open(summary_file, 'w') as f:
        f.write(summary_content)
    
    logger.info(f"Created dataset summary: {summary_file}")
    logger.info("Unsplit dataset creation completed successfully!")
    
    return target_dir

if __name__ == "__main__":
    create_unsplit_dataset() 