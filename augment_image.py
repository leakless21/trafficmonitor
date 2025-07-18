#!/usr/bin/env python3
"""
Image Augmentation Script
Reads augmentation parameters from Augmentations.csv and applies each augmentation to sample images.
"""

import cv2
import numpy as np
import pandas as pd
import os
import math
import random
from pathlib import Path
import albumentations as A
from PIL import Image, ImageDraw, ImageFont

def create_sample_image(width=640, height=640):
    """Create a sample image with geometric patterns for testing augmentations."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add colorful geometric patterns
    # Background gradient
    for y in range(height):
        for x in range(width):
            img[y, x] = [
                int(255 * x / width),
                int(255 * y / height), 
                int(255 * (x + y) / (width + height))
            ]
    
    # Add some geometric shapes
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
    cv2.circle(img, (400, 150), 80, (0, 255, 0), -1)
    cv2.ellipse(img, (500, 400), (100, 60), 45, 0, 360, (0, 0, 255), -1)
    
    # Add text
    cv2.putText(img, "SAMPLE", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(img, "IMAGE", (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    return img

def apply_geometric_augmentation(img, param_name, value):
    """Apply geometric augmentations."""
    h, w = img.shape[:2]
    
    if param_name == "degrees":
        # More drastic rotation - multiply by 5x
        angle = random.uniform(-float(value) * 5, float(value) * 2)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h))
    
    elif param_name == "translate":
        # More drastic translation - multiply by 3x
        translate_percent = float(value) * 3
        tx = random.uniform(-translate_percent, translate_percent) * w
        ty = random.uniform(-translate_percent, translate_percent) * h
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, matrix, (w, h))
    
    elif param_name == "scale":
        # Much more drastic scaling - multiply by 5x
        scale_range = float(value) * 5
        scale_factor = 1.0 + random.uniform(-scale_range, scale_range)
        # Ensure minimum scale to avoid invisibly small images
        scale_factor = max(0.1, min(3.0, scale_factor))
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 0, scale_factor)
        return cv2.warpAffine(img, matrix, (w, h))
    
    elif param_name == "shear":
        # Much more drastic shear - multiply by 10x
        shear_angle = random.uniform(-float(value) * 10, float(value) * 10)
        shear_x = math.tan(math.radians(shear_angle))
        # Also add vertical shear for more dramatic effect
        shear_y = math.tan(math.radians(random.uniform(-float(value) * 5, float(value) * 5)))
        matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
        return cv2.warpAffine(img, matrix, (w, h))
    
    elif param_name == "perspective":
        # Dramatically visible perspective transformation
        # Ignore the tiny CSV value and use a visible range
        perspective_strength = 0.3  # Much stronger base value
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Create extreme perspective distortion with large offsets
        max_offset = min(w, h) * perspective_strength
        dst_points = np.float32([
            [random.uniform(-max_offset, max_offset*0.5), random.uniform(-max_offset*0.5, max_offset)],
            [w + random.uniform(-max_offset*0.5, max_offset), random.uniform(-max_offset, max_offset*0.5)],
            [w + random.uniform(-max_offset, max_offset*0.5), h + random.uniform(-max_offset*0.5, max_offset)],
            [random.uniform(-max_offset*0.5, max_offset), h + random.uniform(-max_offset, max_offset*0.5)]
        ])
        
        try:
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            return cv2.warpPerspective(img, matrix, (w, h))
        except cv2.error:
            # If transformation fails, return original
            return img
    
    elif param_name == "fliplr":
        # Horizontal flip
        if random.random() < float(value):
            return cv2.flip(img, 1)
        else:
            return img.copy()
    
    return img

def apply_color_augmentation(img, param_name, value):
    """Apply color augmentations."""
    if param_name == "hsv_h":
        # More drastic hue shift - multiply by 10x
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_shift = random.uniform(-float(value) * 10, float(value) * 10) * 179  # OpenCV hue range is 0-179
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    elif param_name == "hsv_s":
        # More drastic saturation shift - multiply by 1.5x
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_range = float(value) * 1.5
        sat_factor = 1.0 + random.uniform(-sat_range, sat_range)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    elif param_name == "hsv_v":
        # More drastic brightness shift - multiply by 1.5x
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        val_range = float(value) * 1.5
        val_factor = 1.0 + random.uniform(-val_range, val_range)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return img

def create_mosaic(imgs):
    """Create a mosaic from 4 images."""
    if len(imgs) < 4:
        # If we don't have 4 images, duplicate the first one
        while len(imgs) < 4:
            imgs.append(imgs[0])
    
    h, w = imgs[0].shape[:2]
    # Resize each image to half size
    resized_imgs = [cv2.resize(img, (w//2, h//2)) for img in imgs[:4]]
    
    # Create mosaic
    top_row = np.hstack([resized_imgs[0], resized_imgs[1]])
    bottom_row = np.hstack([resized_imgs[2], resized_imgs[3]])
    mosaic = np.vstack([top_row, bottom_row])
    
    return mosaic

def apply_albumentations(img, param_name, value):
    """Apply Albumentations augmentations."""
    # Parse probability from value string like "p=0.01"
    if "p=" in value:
        prob = float(value.split("p=")[1])
    else:
        prob = 0.5  # default probability
    
    if param_name == "Blur":
        # More drastic blur
        transform = A.Blur(blur_limit=21, p=1.0)  # Always apply, much stronger blur
    elif param_name == "MedianBlur":
        # More drastic median blur
        transform = A.MedianBlur(blur_limit=15, p=1.0)  # Always apply, stronger blur
    elif param_name == "ToGray":
        # Always convert to grayscale
        transform = A.ToGray(p=1.0)
    elif param_name == "CLAHE":
        # More aggressive CLAHE
        transform = A.CLAHE(clip_limit=8.0, tile_grid_size=(4, 4), p=1.0)  # Always apply, stronger contrast
    else:
        return img
    
    # Convert BGR to RGB for Albumentations
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = transform(image=img_rgb)['image']
    # Convert back to BGR
    return cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

def main():
    # Create output directory
    output_dir = Path("augmented_images")
    output_dir.mkdir(exist_ok=True)
    
    # Read augmentation parameters
    try:
        df = pd.read_csv("Augmentations.csv")
        print(f"Loaded {len(df)} augmentation parameters from Augmentations.csv")
    except FileNotFoundError:
        print("Error: Augmentations.csv not found!")
        return
    
    # Load the demo image
    try:
        sample_img = cv2.imread("DemoAugment.jpg")
        if sample_img is None:
            print("Error: DemoAugment.jpg not found! Creating sample image instead.")
            sample_img = create_sample_image()
        else:
            print("Loaded DemoAugment.jpg")
    except Exception as e:
        print(f"Error loading DemoAugment.jpg: {e}. Creating sample image instead.")
        sample_img = create_sample_image()
    
    cv2.imwrite(str(output_dir / "00_original.jpg"), sample_img)
    print("Saved original image: 00_original.jpg")
    
    # Apply each augmentation
    current_category = ""
    for idx, row in df.iterrows():
        category = row['Category']
        parameter = row['Parameter']
        value = row['Value']
        description = row['Description']
        
        # Handle empty category (continuation rows)
        if pd.isna(category) or category == "":
            category = current_category
        else:
            current_category = category
        
        # Skip empty parameter names (continuation rows)
        if pd.isna(parameter) or parameter == "":
            continue
        
        print(f"Applying {category} - {parameter}: {description}")
        
        try:
            if category == "Geometric":
                augmented_img = apply_geometric_augmentation(sample_img.copy(), parameter, value)
            elif category == "Color":
                augmented_img = apply_color_augmentation(sample_img.copy(), parameter, value)
            elif category == "Structural":
                if parameter == "mosaic":
                    # Create mosaic with 4 copies of the sample image
                    augmented_img = create_mosaic([sample_img] * 4)
                else:
                    continue  # Skip close_mosaic as it's just a training parameter
            elif category == "Albumentations":
                augmented_img = apply_albumentations(sample_img.copy(), parameter, value)
            else:
                print(f"Unknown category: {category}")
                continue
            
            # Save augmented image
            filename = f"{idx+1:02d}_{category}_{parameter}.jpg"
            cv2.imwrite(str(output_dir / filename), augmented_img)
            print(f"  Saved: {filename}")
            
        except Exception as e:
            print(f"  Error applying {parameter}: {e}")
    
    print(f"\nAugmentation complete! Check the '{output_dir}' folder for results.")
    print("Note: Some augmentations use random values, so results may vary between runs.")

if __name__ == "__main__":
    main() 