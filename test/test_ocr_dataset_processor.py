#!/usr/bin/env python3
"""
Unit tests for OCR Dataset Processor.
"""

import pytest
import tempfile
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2

# Import functions from the script
import sys
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from ocr_dataset_processor import (
    OCRResult,
    BaseOCREngine,
    FastPlateOCREngine,
    PaddleOCREngine,
    get_ocr_engine,
    get_image_files,
    process_dataset
)


class TestOCRResult:
    """Test OCRResult dataclass."""
    
    def test_ocr_result_creation(self):
        """Test OCRResult creation."""
        result = OCRResult(
            text="ABC123",
            confidence=0.95,
            processing_time=0.1234,
            engine="test_engine"
        )
        
        assert result.text == "ABC123"
        assert result.confidence == 0.95
        assert result.processing_time == 0.1234
        assert result.engine == "test_engine"


class TestBaseOCREngine:
    """Test BaseOCREngine class."""
    
    def test_base_engine_init(self):
        """Test BaseOCREngine initialization."""
        config = {"param1": "value1"}
        engine = BaseOCREngine(config)
        
        assert engine.config == config
        assert engine.name == "base"
    
    def test_base_engine_process_image_not_implemented(self):
        """Test that process_image raises NotImplementedError."""
        engine = BaseOCREngine({})
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(NotImplementedError):
            engine.process_image(image)


class MockLicensePlateRecognizer:
    """Mock for LicensePlateRecognizer."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def run(self, image, return_confidence=True):
        # Simulate successful OCR
        plate_texts = ["ABC123"]
        confidence = np.array([[0.9, 0.8, 0.7, 0.9, 0.8, 0.7]])
        return (plate_texts, confidence)


class TestFastPlateOCREngine:
    """Test FastPlateOCREngine class."""
    
    @patch('ocr_dataset_processor.LicensePlateRecognizer', new=MockLicensePlateRecognizer)
    def test_fast_plate_ocr_init_success(self):
        """Test successful FastPlateOCREngine initialization."""
        config = {
            "hub_model_name": "test-model",
            "device": "cpu",
            "conf_threshold": 0.6
        }
        
        engine = FastPlateOCREngine(config)
        assert engine.name == "fast_plate_ocr"
        assert engine.conf_threshold == 0.6
    
    @patch('ocr_dataset_processor.LicensePlateRecognizer', side_effect=ImportError("Module not found"))
    def test_fast_plate_ocr_import_error(self, mock_recognizer):
        """Test ImportError handling."""
        config = {}
        
        with pytest.raises(ImportError, match="fast_plate_ocr is not installed"):
            FastPlateOCREngine(config)
    
    @patch('ocr_dataset_processor.LicensePlateRecognizer', new=MockLicensePlateRecognizer)
    def test_fast_plate_ocr_process_image_success(self):
        """Test successful image processing."""
        config = {"conf_threshold": 0.5}
        engine = FastPlateOCREngine(config)
        
        # Create a test image
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        result = engine.process_image(image)
        
        assert result is not None
        assert isinstance(result, OCRResult)
        assert result.text == "ABC123"
        assert result.engine == "fast_plate_ocr"
        assert result.confidence > 0.5
        assert result.processing_time >= 0
    
    @patch('ocr_dataset_processor.LicensePlateRecognizer', new=MockLicensePlateRecognizer)
    def test_fast_plate_ocr_preprocess_image(self):
        """Test image preprocessing (no preprocessing needed for new API)."""
        config = {}
        engine = FastPlateOCREngine(config)
        
        # Test color image - should return original image
        color_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = engine._preprocess_image(color_image)
        assert np.array_equal(result, color_image)  # Should be unchanged
        
        # Test already grayscale image
        gray_image = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        result = engine._preprocess_image(gray_image)
        assert np.array_equal(result, gray_image)


class MockPaddleOCR:
    """Mock for PaddleOCR."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def ocr(self, image, cls=True):
        # Simulate PaddleOCR v3.x response format
        return [
            [
                [[0, 0], [100, 0], [100, 30], [0, 30]], 
                ("ABC123", 0.95)
            ]
        ]


class TestPaddleOCREngine:
    """Test PaddleOCREngine class."""
    
    @patch('ocr_dataset_processor.PaddleOCR', new=MockPaddleOCR)
    def test_paddleocr_init_success(self):
        """Test successful PaddleOCREngine initialization."""
        config = {
            "lang": "en",
            "use_gpu": False,
            "conf_threshold": 0.7
        }
        
        engine = PaddleOCREngine(config)
        assert engine.name == "paddleocr"
        assert engine.conf_threshold == 0.7
    
    @patch('ocr_dataset_processor.PaddleOCR', side_effect=ImportError("Module not found"))
    def test_paddleocr_import_error(self, mock_paddle):
        """Test ImportError handling."""
        config = {}
        
        with pytest.raises(ImportError, match="PaddleOCR is not installed"):
            PaddleOCREngine(config)
    
    @patch('ocr_dataset_processor.PaddleOCR', new=MockPaddleOCR)
    def test_paddleocr_process_image_success(self):
        """Test successful image processing."""
        config = {"conf_threshold": 0.5}
        engine = PaddleOCREngine(config)
        
        # Create a test image
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        result = engine.process_image(image)
        
        assert result is not None
        assert isinstance(result, OCRResult)
        assert result.text == "ABC123"
        assert result.engine == "paddleocr"
        assert result.confidence == 0.95
        assert result.processing_time >= 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_ocr_engine_fast_plate_ocr(self):
        """Test getting FastPlateOCR engine."""
        with patch('ocr_dataset_processor.FastPlateOCREngine') as mock_engine:
            mock_instance = Mock()
            mock_engine.return_value = mock_instance
            
            config = {"test": "config"}
            result = get_ocr_engine("fast_plate_ocr", config)
            
            mock_engine.assert_called_once_with(config)
            assert result == mock_instance
    
    def test_get_ocr_engine_paddleocr(self):
        """Test getting PaddleOCR engine."""
        with patch('ocr_dataset_processor.PaddleOCREngine') as mock_engine:
            mock_instance = Mock()
            mock_engine.return_value = mock_instance
            
            config = {"test": "config"}
            result = get_ocr_engine("paddleocr", config)
            
            mock_engine.assert_called_once_with(config)
            assert result == mock_instance
    
    def test_get_ocr_engine_invalid(self):
        """Test invalid engine name."""
        with pytest.raises(ValueError, match="Unknown engine: invalid"):
            get_ocr_engine("invalid", {})
    
    def test_get_image_files(self):
        """Test getting image files from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test1.jpg").touch()
            (temp_path / "test2.PNG").touch()
            (temp_path / "test3.jpeg").touch()
            (temp_path / "test4.txt").touch()  # Should be ignored
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "test5.bmp").touch()
            
            image_files = get_image_files(temp_path)
            
            # Should find 4 image files
            assert len(image_files) == 4
            
            # Check extensions
            extensions = {f.suffix.lower() for f in image_files}
            assert extensions == {'.jpg', '.png', '.jpeg', '.bmp'}


class TestProcessDataset:
    """Test process_dataset function."""
    
    @patch('ocr_dataset_processor.get_ocr_engine')
    @patch('ocr_dataset_processor.get_image_files')
    @patch('cv2.imread')
    def test_process_dataset_success(self, mock_imread, mock_get_files, mock_get_engine):
        """Test successful dataset processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_csv = temp_path / "output.csv"
            
            # Mock dependencies
            mock_engine = Mock()
            mock_get_engine.return_value = mock_engine
            
            # Mock image files
            image_files = [
                temp_path / "img1.jpg",
                temp_path / "img2.jpg"
            ]
            mock_get_files.return_value = image_files
            
            # Mock cv2.imread
            mock_imread.return_value = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            
            # Mock OCR results
            ocr_results = [
                OCRResult("ABC123", 0.95, 0.1, "test_engine"),
                OCRResult("XYZ789", 0.90, 0.15, "test_engine")
            ]
            mock_engine.process_image.side_effect = ocr_results
            
            # Run processing
            stats = process_dataset(
                input_dir=temp_path,
                output_csv=output_csv,
                engine_name="fast_plate_ocr",
                engine_config={"test": "config"},
                relative_paths=True
            )
            
            # Verify results
            assert stats["total_images"] == 2
            assert stats["successful"] == 2
            assert stats["failed"] == 0
            assert stats["success_rate"] == 1.0
            assert stats["engine"] == "fast_plate_ocr"
            
            # Verify CSV was created
            assert output_csv.exists()
            
            # Read and verify CSV contents
            with open(output_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Check header and data
            assert len(rows) == 3  # Header + 2 data rows
            assert rows[0] == ['image_path', 'plate_text', 'confidence', 'processing_time', 'engine']
            assert rows[1][1] == 'ABC123'
            assert rows[2][1] == 'XYZ789'
    
    @patch('ocr_dataset_processor.get_ocr_engine')
    @patch('ocr_dataset_processor.get_image_files')
    def test_process_dataset_no_images(self, mock_get_files, mock_get_engine):
        """Test processing with no image files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_csv = temp_path / "output.csv"
            
            # Mock no image files
            mock_get_files.return_value = []
            
            stats = process_dataset(
                input_dir=temp_path,
                output_csv=output_csv,
                engine_name="fast_plate_ocr",
                engine_config={}
            )
            
            assert stats["total_images"] == 0
            assert stats["processed"] == 0
            assert stats["successful"] == 0
    
    @patch('ocr_dataset_processor.get_ocr_engine')
    def test_process_dataset_engine_init_failure(self, mock_get_engine):
        """Test processing with engine initialization failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_csv = temp_path / "output.csv"
            
            # Mock engine initialization failure
            mock_get_engine.side_effect = RuntimeError("Engine init failed")
            
            with pytest.raises(RuntimeError, match="Engine init failed"):
                process_dataset(
                    input_dir=temp_path,
                    output_csv=output_csv,
                    engine_name="fast_plate_ocr",
                    engine_config={}
                )


if __name__ == "__main__":
    pytest.main([__file__]) 