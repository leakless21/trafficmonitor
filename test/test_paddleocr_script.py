#!/usr/bin/env python3
"""
Unit tests for PaddleOCR to CSV script.
"""

import pytest
import tempfile
import csv
from pathlib import Path
from unittest.mock import Mock, patch

# Import functions from the script
import sys
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from paddleocr_to_csv import (
    initialize_ocr,
    process_image,
    get_image_files,
    process_images_to_csv
)


class TestPaddleOCRScript:
    """Test cases for PaddleOCR script functions."""

    def test_get_image_files(self):
        """Test getting image files from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test1.jpg").touch()
            (temp_path / "test2.PNG").touch()
            (temp_path / "test3.txt").touch()  # Should be ignored
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "test4.jpeg").touch()
            
            image_files = get_image_files(temp_path)
            
            # Should find 3 image files (jpg, PNG, jpeg)
            assert len(image_files) == 3
            
            # Check file extensions
            extensions = {f.suffix.lower() for f in image_files}
            assert extensions == {'.jpg', '.png', '.jpeg'}

    @patch('paddleocr_to_csv.PaddleOCR')
    def test_initialize_ocr(self, mock_paddle_ocr):
        """Test OCR initialization."""
        mock_ocr_instance = Mock()
        mock_paddle_ocr.return_value = mock_ocr_instance
        
        ocr = initialize_ocr(use_gpu=True, lang='en')
        
        # Verify PaddleOCR was called with correct parameters
        mock_paddle_ocr.assert_called_once_with(
            use_angle_cls=True,
            lang='en',
            use_gpu=True,
            show_log=False
        )
        
        assert ocr == mock_ocr_instance

    @patch('cv2.imread')
    def test_process_image_success(self, mock_imread):
        """Test successful image processing."""
        # Mock image data
        mock_imread.return_value = Mock()  # Simulated image
        
        # Mock OCR instance and result
        mock_ocr = Mock()
        mock_ocr.ocr.return_value = [[
            [[[0, 0], [100, 0], [100, 30], [0, 30]], ('ABC123', 0.95)]
        ]]
        
        result = process_image(Path("test.jpg"), mock_ocr)
        
        assert result == "ABC123"
        mock_imread.assert_called_once()
        mock_ocr.ocr.assert_called_once()

    @patch('cv2.imread')
    def test_process_image_no_text(self, mock_imread):
        """Test image processing with no text detected."""
        mock_imread.return_value = Mock()
        
        # Mock OCR with no results
        mock_ocr = Mock()
        mock_ocr.ocr.return_value = [[]]
        
        result = process_image(Path("test.jpg"), mock_ocr)
        
        assert result is None

    @patch('cv2.imread')
    def test_process_image_low_confidence(self, mock_imread):
        """Test image processing with low confidence text."""
        mock_imread.return_value = Mock()
        
        # Mock OCR with low confidence result
        mock_ocr = Mock()
        mock_ocr.ocr.return_value = [[
            [[[0, 0], [100, 0], [100, 30], [0, 30]], ('ABC123', 0.3)]  # Low confidence
        ]]
        
        result = process_image(Path("test.jpg"), mock_ocr)
        
        assert result is None

    def test_process_image_file_not_found(self):
        """Test processing non-existent image file."""
        mock_ocr = Mock()
        
        result = process_image(Path("non_existent.jpg"), mock_ocr)
        
        assert result is None

    @patch('paddleocr_to_csv.initialize_ocr')
    @patch('paddleocr_to_csv.get_image_files')
    @patch('paddleocr_to_csv.process_image')
    def test_process_images_to_csv(self, mock_process_image, mock_get_files, mock_init_ocr):
        """Test complete CSV processing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_csv = temp_path / "output.csv"
            
            # Mock dependencies
            mock_ocr = Mock()
            mock_init_ocr.return_value = mock_ocr
            
            # Mock image files
            image_files = [
                temp_path / "img1.jpg",
                temp_path / "img2.jpg"
            ]
            mock_get_files.return_value = image_files
            
            # Mock OCR results
            mock_process_image.side_effect = ["ABC123", "XYZ789"]
            
            # Run processing
            process_images_to_csv(
                input_dir=temp_path,
                output_csv=output_csv,
                use_gpu=False,
                lang='en'
            )
            
            # Verify CSV was created
            assert output_csv.exists()
            
            # Read and verify CSV contents
            with open(output_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Check header and data
            assert len(rows) == 3  # Header + 2 data rows
            assert rows[0] == ['image_path', 'plate_text']
            assert rows[1][1] == 'ABC123'
            assert rows[2][1] == 'XYZ789'


if __name__ == "__main__":
    pytest.main([__file__]) 