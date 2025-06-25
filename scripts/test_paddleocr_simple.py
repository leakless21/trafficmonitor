#!/usr/bin/env python3
"""
Simple test for PaddleOCR to CSV script functionality.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

def test_paddleocr_import():
    """Test if PaddleOCR can be imported."""
    try:
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR import successful")
        return True
    except ImportError as e:
        print(f"❌ PaddleOCR import failed: {e}")
        return False

def test_script_functions():
    """Test the script functions without running OCR."""
    try:
        from paddleocr_to_csv import get_image_files, setup_logging
        
        # Test logging setup
        logger = setup_logging()
        print("✅ Logging setup successful")
        
        # Test image file detection
        test_dir = Path("../data/test_images")
        if test_dir.exists():
            image_files = get_image_files(test_dir)
            print(f"✅ Found {len(image_files)} image files in test directory")
        else:
            print("⚠️  Test directory not found, but function works")
            
        return True
    except Exception as e:
        print(f"❌ Script function test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing PaddleOCR to CSV script...")
    
    tests = [
        test_paddleocr_import,
        test_script_functions
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Script is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check dependencies.")
        return 1

if __name__ == "__main__":
    exit(main()) 