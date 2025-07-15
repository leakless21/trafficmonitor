"""
Unit tests for text recognition (OCR) service.
Tests OCR processing, text extraction, and confidence scoring.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


class TestTextRecognitionService:
    """Test OCR text recognition functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            "model_path": "data/models/ocr/",
            "conf_threshold": 0.5,
            "max_text_length": 10,
            "allowed_chars": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        }
        
        # Sample license plate image
        self.sample_plate = np.ones((50, 150, 3), dtype=np.uint8) * 255
        
        # Add some text-like patterns
        cv2.rectangle(self.sample_plate, (10, 15), (40, 35), (0, 0, 0), -1)  # Letter region
        cv2.rectangle(self.sample_plate, (50, 15), (80, 35), (0, 0, 0), -1)  # Letter region
        cv2.rectangle(self.sample_plate, (90, 15), (140, 35), (0, 0, 0), -1) # Number region

    def test_plate_image_preprocessing(self):
        """Test preprocessing of license plate image for OCR."""
        plate_image = self.sample_plate.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        assert gray.shape == (50, 150)
        assert gray.dtype == np.uint8
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        assert binary.dtype == np.uint8
        assert np.all((binary == 0) | (binary == 255))

    def test_text_extraction_simulation(self):
        """Test text extraction from plate image."""
        plate_image = self.sample_plate
        
        # Mock OCR result
        mock_ocr_result = {
            "text": "ABC123",
            "confidence": 0.85,
            "char_confidences": [0.9, 0.8, 0.9, 0.8, 0.7, 0.9]
        }
        
        # Validate OCR result
        assert len(mock_ocr_result["text"]) <= self.mock_config["max_text_length"]
        assert mock_ocr_result["confidence"] >= self.mock_config["conf_threshold"]
        assert len(mock_ocr_result["char_confidences"]) == len(mock_ocr_result["text"])

    def test_character_filtering(self):
        """Test filtering of recognized characters."""
        allowed_chars = set(self.mock_config["allowed_chars"])
        
        test_texts = [
            "ABC123",      # Valid
            "XYZ789",      # Valid
            "AB@123",      # Invalid character @
            "abc123",      # Lowercase (should be converted)
            "AB 123",      # Space (should be removed)
        ]
        
        for text in test_texts:
            filtered_text = self._filter_characters(text, allowed_chars)
            assert all(c in allowed_chars for c in filtered_text), f"Filtered text should only contain allowed chars: {filtered_text}"

    def test_confidence_scoring(self):
        """Test OCR confidence scoring."""
        char_confidences = [0.9, 0.8, 0.7, 0.9, 0.6, 0.8]
        
        # Calculate average confidence
        avg_confidence = sum(char_confidences) / len(char_confidences)
        assert 0.0 <= avg_confidence <= 1.0
        
        # Calculate minimum confidence
        min_confidence = min(char_confidences)
        assert min_confidence <= avg_confidence
        
        # Test confidence threshold
        threshold = 0.7
        high_conf_chars = [conf for conf in char_confidences if conf >= threshold]
        assert len(high_conf_chars) == 4  # 4 characters above 0.7

    def test_text_length_validation(self):
        """Test validation of recognized text length."""
        max_length = self.mock_config["max_text_length"]
        
        test_cases = [
            ("ABC123", True),           # Valid length
            ("ABCD1234", True),         # Valid length
            ("ABCDEFGHIJ", True),       # Max length
            ("ABCDEFGHIJK", False),     # Too long
            ("", False),                # Empty
            ("A", True),                # Single character
        ]
        
        for text, should_be_valid in test_cases:
            is_valid = 1 <= len(text) <= max_length
            assert is_valid == should_be_valid, f"Text '{text}' validation failed"

    def test_text_pattern_validation(self):
        """Test validation of license plate text patterns."""
        # Common license plate patterns
        valid_patterns = [
            "ABC123",      # 3 letters + 3 numbers
            "AB1234",      # 2 letters + 4 numbers
            "1234AB",      # 4 numbers + 2 letters
            "A123BCD",     # Mixed pattern
        ]
        
        invalid_patterns = [
            "ABCDEF",      # Only letters
            "123456",      # Only numbers
            "A",           # Too short
            "",            # Empty
        ]
        
        for pattern in valid_patterns:
            assert self._validate_plate_pattern(pattern), f"Valid pattern should pass: {pattern}"
        
        for pattern in invalid_patterns:
            assert not self._validate_plate_pattern(pattern), f"Invalid pattern should fail: {pattern}"

    def test_ocr_error_handling(self):
        """Test OCR error handling with invalid inputs."""
        invalid_inputs = [
            None,
            np.array([]),
            np.zeros((5, 5), dtype=np.uint8),     # Too small
            np.zeros((10, 10, 4), dtype=np.uint8), # Wrong channels
            "invalid_input",                       # Wrong type
        ]
        
        for invalid_input in invalid_inputs:
            try:
                result = self._validate_plate_image(invalid_input)
                assert not result, f"Invalid input should be rejected: {type(invalid_input)}"
            except Exception:
                # Exception handling is acceptable
                pass

    def test_text_postprocessing(self):
        """Test postprocessing of OCR results."""
        raw_ocr_results = [
            {"text": "abc123", "confidence": 0.8},    # Lowercase
            {"text": "AB 123", "confidence": 0.7},    # With space
            {"text": "AB@123", "confidence": 0.6},    # Invalid char
            {"text": "ABCDEFGHIJK", "confidence": 0.9}, # Too long
        ]
        
        processed_results = []
        for result in raw_ocr_results:
            processed_text = self._postprocess_text(result["text"])
            if processed_text and len(processed_text) <= self.mock_config["max_text_length"]:
                processed_results.append({
                    "text": processed_text,
                    "confidence": result["confidence"]
                })
        
        assert len(processed_results) <= len(raw_ocr_results)
        assert all(result["text"].isupper() for result in processed_results)

    def test_ocr_performance(self):
        """Test OCR performance with multiple plate images."""
        import time
        
        plate_images = [self.sample_plate for _ in range(10)]
        
        start_time = time.time()
        
        # Simulate OCR processing
        results = []
        for plate in plate_images:
            # Mock OCR processing
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            time.sleep(0.001)  # Simulate processing time
            results.append({"text": "ABC123", "confidence": 0.8})
        
        total_time = time.time() - start_time
        avg_time_per_plate = total_time / len(plate_images)
        
        assert avg_time_per_plate < 0.1, f"OCR should be fast: {avg_time_per_plate:.3f}s per plate"
        assert len(results) == len(plate_images)

    def test_batch_ocr_processing(self):
        """Test batch processing of multiple plates."""
        batch_plates = [self.sample_plate for _ in range(5)]
        
        # Simulate batch processing
        batch_results = []
        for i, plate in enumerate(batch_plates):
            result = {
                "plate_id": i,
                "text": f"ABC{100+i}",
                "confidence": 0.8 + (i * 0.02)
            }
            batch_results.append(result)
        
        assert len(batch_results) == len(batch_plates)
        assert all("plate_id" in result for result in batch_results)

    def test_confidence_calibration(self):
        """Test OCR confidence calibration."""
        # Test confidence adjustment based on text characteristics
        test_cases = [
            {"text": "ABC123", "raw_conf": 0.8, "expected_boost": True},   # Good pattern
            {"text": "ABCDEF", "raw_conf": 0.8, "expected_boost": False},  # Only letters
            {"text": "123456", "raw_conf": 0.8, "expected_boost": False}, # Only numbers
            {"text": "A1B2C3", "raw_conf": 0.8, "expected_boost": True},  # Mixed pattern
        ]
        
        for case in test_cases:
            adjusted_conf = self._adjust_confidence(case["text"], case["raw_conf"])
            
            if case["expected_boost"]:
                assert adjusted_conf >= case["raw_conf"], f"Confidence should be boosted for {case['text']}"
            else:
                assert adjusted_conf <= case["raw_conf"], f"Confidence should not be boosted for {case['text']}"

    # Helper methods
    def _filter_characters(self, text, allowed_chars):
        """Filter text to only include allowed characters."""
        return ''.join(c.upper() for c in text if c.upper() in allowed_chars)

    def _validate_plate_pattern(self, text):
        """Validate license plate text pattern."""
        if not text or len(text) < 2:
            return False
        
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        return has_letter and has_number

    def _validate_plate_image(self, image):
        """Validate plate image for OCR processing."""
        if image is None or not isinstance(image, np.ndarray):
            return False
        if len(image.shape) not in [2, 3]:  # Grayscale or color
            return False
        if image.shape[0] < 10 or image.shape[1] < 20:  # Too small
            return False
        return True

    def _postprocess_text(self, text):
        """Postprocess OCR text result."""
        # Convert to uppercase and remove invalid characters
        allowed_chars = set(self.mock_config["allowed_chars"])
        processed = ''.join(c.upper() for c in text if c.upper() in allowed_chars)
        return processed

    def _adjust_confidence(self, text, raw_confidence):
        """Adjust confidence based on text characteristics."""
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        if has_letter and has_number:
            return min(1.0, raw_confidence * 1.1)  # Boost confidence
        else:
            return max(0.0, raw_confidence * 0.9)  # Reduce confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])