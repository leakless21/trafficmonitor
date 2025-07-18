#!/usr/bin/env python3
"""
Combined script to generate augmented images and create a collage.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run both augmentation and collage generation."""
    
    print("Step 1: Generating augmented images...")
    try:
        result = subprocess.run([sys.executable, "augment_image.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error generating augmented images: {e}")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        return
    
    print("\nStep 2: Creating collage...")
    try:
        result = subprocess.run([sys.executable, "create_collage.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error creating collage: {e}")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        return
    
    print("\n✅ Complete! Check these files:")
    print("📁 augmented_images/ - Individual augmented images")
    print("🖼️  augmentation_collage.jpg - Combined view (JPG)")
    print("🖼️  augmentation_collage_hq.png - Combined view (PNG, high quality)")

if __name__ == "__main__":
    main() 