#!/usr/bin/env python3
"""
Create a collage of all augmented images for easy comparison.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd

def resize_image(img, target_size=(450, 450)):
    """Resize image to fill target size with minimal padding."""
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling to fill the target size (crop if needed)
    scale_w = target_w / w
    scale_h = target_h / h
    scale = max(scale_w, scale_h)  # Use larger scale to fill completely
    
    # Resize image
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Crop to exact target size from center
    if new_w > target_w:
        x_offset = (new_w - target_w) // 2
        resized = resized[:, x_offset:x_offset + target_w]
    
    if new_h > target_h:
        y_offset = (new_h - target_h) // 2
        resized = resized[y_offset:y_offset + target_h, :]
    
    # If somehow smaller, pad with white
    if resized.shape[:2] != (target_h, target_w):
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        h_r, w_r = resized.shape[:2]
        y_offset = (target_h - h_r) // 2
        x_offset = (target_w - w_r) // 2
        canvas[y_offset:y_offset+h_r, x_offset:x_offset+w_r] = resized
        return canvas
    
    return resized

def add_text_label(img, text, position='bottom'):
    """Add simple text label to image with subtle background."""
    h, w = img.shape[:2]
    
    # Create a copy to modify
    labeled_img = img.copy()
    
    # Slightly larger font for better readability at higher resolution
    font_scale = 0.6
    thickness = 1
    color = (0, 0, 0)  # Black text
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Position text
    if position == 'bottom':
        text_y = h - 15
        # Add subtle white background for readability
        cv2.rectangle(labeled_img, (0, h - text_h - 20), (w, h), (255, 255, 255), -1)
        cv2.rectangle(labeled_img, (0, h - text_h - 20), (w, h), (200, 200, 200), 1)
    else:  # top
        text_y = text_h + 15
        cv2.rectangle(labeled_img, (0, 0), (w, text_h + 20), (255, 255, 255), -1)
        cv2.rectangle(labeled_img, (0, 0), (w, text_h + 20), (200, 200, 200), 1)
    
    # Center text horizontally
    text_x = (w - text_w) // 2
    
    cv2.putText(labeled_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness, cv2.LINE_AA)
    
    return labeled_img

def create_collage():
    """Create a collage of all augmented images."""
    
    # Get all augmented images
    augmented_dir = Path("augmented_images")
    if not augmented_dir.exists():
        print("Error: augmented_images directory not found!")
        return
    
    # Load CSV to get descriptions
    try:
        df = pd.read_csv("Augmentations.csv")
        descriptions = {}
        current_category = ""
        
        for _, row in df.iterrows():
            category = row['Category']
            parameter = row['Parameter']
            description = row['Description']
            
            # Handle empty category (continuation rows)
            if pd.isna(category) or category == "":
                category = current_category
            else:
                current_category = category
            
            if not (pd.isna(parameter) or parameter == ""):
                # Create simple, short labels
                if category == "Geometric":
                    descriptions[parameter] = parameter.capitalize()
                elif category == "Color":
                    descriptions[parameter] = parameter.upper()
                elif category == "Structural":
                    descriptions[parameter] = parameter.capitalize()
                elif category == "Albumentations":
                    descriptions[parameter] = parameter
                else:
                    descriptions[parameter] = parameter
                
    except FileNotFoundError:
        descriptions = {}
    
    # Get all image files and sort them
    image_files = sorted([f for f in augmented_dir.glob("*.jpg")])
    
    if not image_files:
        print("No images found in augmented_images directory!")
        return
    
    print(f"Found {len(image_files)} images to include in collage")
    
    # Load and process images
    processed_images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        # Resize to larger size for better quality
        resized_img = resize_image(img, (450, 450))
        
        # Extract label from filename
        filename = img_file.stem
        if filename.startswith("00_"):
            label = "Original"
        else:
            # Parse filename like "01_Geometric_degrees.jpg"
            parts = filename.split("_", 2)
            if len(parts) >= 3:
                param_name = parts[2]
                label = descriptions.get(param_name, param_name)
            else:
                label = filename
        
        # Add label to image
        labeled_img = add_text_label(resized_img, label)
        processed_images.append(labeled_img)
    
    # Calculate grid dimensions
    num_images = len(processed_images)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    print(f"Creating {rows}x{cols} grid for {num_images} images")
    
    # Create collage with white background
    img_h, img_w = processed_images[0].shape[:2]
    collage_h = rows * img_h
    collage_w = cols * img_w
    collage = np.ones((collage_h, collage_w, 3), dtype=np.uint8) * 255
    
    # Fill the collage
    for i, img in enumerate(processed_images):
        row = i // cols
        col = i % cols
        
        y_start = row * img_h
        y_end = y_start + img_h
        x_start = col * img_w
        x_end = x_start + img_w
        
        collage[y_start:y_end, x_start:x_end] = img
    
    # Save high-quality collage
    output_path_hq = "augmentation_collage_hq.png"
    cv2.imwrite(output_path_hq, collage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"High-quality collage saved as: {output_path_hq}")
    
    # Also create a compressed version for sharing
    output_path = "augmentation_collage.jpg"
    cv2.imwrite(output_path, collage, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Compressed collage saved as: {output_path}")
    
    print(f"Collage dimensions: {collage_w}x{collage_h} pixels")
    print(f"Individual image size: 450x450 pixels")
    print(f"Total area: {(collage_w * collage_h) / 1000000:.1f} megapixels")

if __name__ == "__main__":
    create_collage() 