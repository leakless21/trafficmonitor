"""
Unit tests for utils module.
Tests coordinate conversion functions and utility helpers.
"""

import pytest
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.utils.utils import relative_to_absolute_coords, absolute_to_relative_coords


class TestUtils:
    """Test utility functions for coordinate conversion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Standard frame dimensions
        self.frame_width = 1920
        self.frame_height = 1080
        
        # Sample relative coordinates (0.0-1.0)
        self.relative_lines = [
            [[0.1, 0.2], [0.9, 0.8]],  # Diagonal line
            [[0.0, 0.5], [1.0, 0.5]],  # Horizontal line
            [[0.5, 0.0], [0.5, 1.0]]   # Vertical line
        ]
        
        # Expected absolute coordinates
        self.expected_absolute = [
            [[192, 216], [1728, 864]],  # Diagonal line
            [[0, 540], [1920, 540]],    # Horizontal line
            [[960, 0], [960, 1080]]     # Vertical line
        ]
        
        # Sample absolute coordinates
        self.absolute_lines = [
            [[100, 200], [800, 600]],   # Diagonal line
            [[0, 540], [1920, 540]],    # Horizontal line
            [[960, 0], [960, 1080]]     # Vertical line
        ]
        
        # Expected relative coordinates
        self.expected_relative = [
            [[100/1920, 200/1080], [800/1920, 600/1080]],  # Diagonal line
            [[0.0, 0.5], [1.0, 0.5]],                       # Horizontal line
            [[0.5, 0.0], [0.5, 1.0]]                        # Vertical line
        ]

    def test_relative_to_absolute_conversion(self):
        """Test conversion from relative to absolute coordinates."""
        result = relative_to_absolute_coords(
            self.relative_lines, 
            self.frame_width, 
            self.frame_height
        )
        
        # Verify conversion
        assert len(result) == len(self.expected_absolute)
        
        for i, line in enumerate(result):
            expected_line = self.expected_absolute[i]
            assert len(line) == len(expected_line)
            
            for j, point in enumerate(line):
                expected_point = expected_line[j]
                assert point[0] == expected_point[0], f"X coordinate mismatch at line {i}, point {j}"
                assert point[1] == expected_point[1], f"Y coordinate mismatch at line {i}, point {j}"

    def test_absolute_to_relative_conversion(self):
        """Test conversion from absolute to relative coordinates."""
        result = absolute_to_relative_coords(
            self.absolute_lines,
            self.frame_width,
            self.frame_height
        )
        
        # Verify conversion
        assert len(result) == len(self.expected_relative)
        
        for i, line in enumerate(result):
            expected_line = self.expected_relative[i]
            assert len(line) == len(expected_line)
            
            for j, point in enumerate(line):
                expected_point = expected_line[j]
                assert abs(point[0] - expected_point[0]) < 1e-6, f"X coordinate mismatch at line {i}, point {j}"
                assert abs(point[1] - expected_point[1]) < 1e-6, f"Y coordinate mismatch at line {i}, point {j}"

    def test_round_trip_conversion(self):
        """Test that converting relative->absolute->relative preserves values."""
        # Start with relative coordinates
        original_relative = self.relative_lines
        
        # Convert to absolute
        absolute_coords = relative_to_absolute_coords(
            original_relative,
            self.frame_width,
            self.frame_height
        )
        
        # Convert back to relative
        result_relative = absolute_to_relative_coords(
            absolute_coords,
            self.frame_width,
            self.frame_height
        )
        
        # Verify round-trip preservation
        assert len(result_relative) == len(original_relative)
        
        for i, line in enumerate(result_relative):
            original_line = original_relative[i]
            assert len(line) == len(original_line)
            
            for j, point in enumerate(line):
                original_point = original_line[j]
                assert abs(point[0] - original_point[0]) < 1e-6, f"Round-trip error at line {i}, point {j}"
                assert abs(point[1] - original_point[1]) < 1e-6, f"Round-trip error at line {i}, point {j}"

    def test_edge_case_coordinates(self):
        """Test edge cases with boundary coordinates."""
        edge_relative = [
            [[0.0, 0.0], [1.0, 1.0]],  # Full diagonal
            [[0.0, 0.0], [0.0, 1.0]],  # Left edge
            [[1.0, 0.0], [1.0, 1.0]],  # Right edge
            [[0.0, 0.0], [1.0, 0.0]],  # Top edge
            [[0.0, 1.0], [1.0, 1.0]]   # Bottom edge
        ]
        
        expected_absolute = [
            [[0, 0], [1920, 1080]],     # Full diagonal
            [[0, 0], [0, 1080]],        # Left edge
            [[1920, 0], [1920, 1080]],  # Right edge
            [[0, 0], [1920, 0]],        # Top edge
            [[0, 1080], [1920, 1080]]   # Bottom edge
        ]
        
        result = relative_to_absolute_coords(
            edge_relative,
            self.frame_width,
            self.frame_height
        )
        
        # Verify edge cases
        for i, line in enumerate(result):
            expected_line = expected_absolute[i]
            for j, point in enumerate(line):
                expected_point = expected_line[j]
                assert point[0] == expected_point[0]
                assert point[1] == expected_point[1]

    def test_single_point_lines(self):
        """Test handling of single-point lines (points)."""
        single_points_relative = [
            [[0.5, 0.5], [0.5, 0.5]],  # Center point
            [[0.0, 0.0], [0.0, 0.0]],  # Top-left corner
            [[1.0, 1.0], [1.0, 1.0]]   # Bottom-right corner
        ]
        
        expected_absolute = [
            [[960, 540], [960, 540]],    # Center point
            [[0, 0], [0, 0]],            # Top-left corner
            [[1920, 1080], [1920, 1080]] # Bottom-right corner
        ]
        
        result = relative_to_absolute_coords(
            single_points_relative,
            self.frame_width,
            self.frame_height
        )
        
        # Verify single points
        for i, line in enumerate(result):
            expected_line = expected_absolute[i]
            for j, point in enumerate(line):
                expected_point = expected_line[j]
                assert point[0] == expected_point[0]
                assert point[1] == expected_point[1]

    def test_different_frame_sizes(self):
        """Test coordinate conversion with different frame sizes."""
        test_cases = [
            (640, 480),    # VGA
            (1280, 720),   # HD
            (1920, 1080),  # Full HD
            (3840, 2160),  # 4K
            (100, 100),    # Square
            (1, 1)         # Minimal
        ]
        
        relative_line = [[[0.25, 0.25], [0.75, 0.75]]]
        
        for width, height in test_cases:
            # Convert to absolute
            absolute_result = relative_to_absolute_coords(relative_line, width, height)
            
            # Verify conversion
            expected_x1 = int(0.25 * width)
            expected_y1 = int(0.25 * height)
            expected_x2 = int(0.75 * width)
            expected_y2 = int(0.75 * height)
            
            assert absolute_result[0][0][0] == expected_x1
            assert absolute_result[0][0][1] == expected_y1
            assert absolute_result[0][1][0] == expected_x2
            assert absolute_result[0][1][1] == expected_y2

    def test_empty_input(self):
        """Test handling of empty input."""
        empty_lines = []
        
        # Test relative to absolute
        result_abs = relative_to_absolute_coords(empty_lines, self.frame_width, self.frame_height)
        assert result_abs == [], "Empty input should return empty output"
        
        # Test absolute to relative
        result_rel = absolute_to_relative_coords(empty_lines, self.frame_width, self.frame_height)
        assert result_rel == [], "Empty input should return empty output"

    def test_precision_preservation(self):
        """Test that precision is preserved in conversions."""
        # Use high-precision relative coordinates
        precise_relative = [
            [[0.123456789, 0.987654321], [0.555555555, 0.333333333]]
        ]
        
        # Convert to absolute and back
        absolute_coords = relative_to_absolute_coords(
            precise_relative,
            self.frame_width,
            self.frame_height
        )
        
        result_relative = absolute_to_relative_coords(
            absolute_coords,
            self.frame_width,
            self.frame_height
        )
        
        # Check precision (allowing for floating-point rounding)
        original_point = precise_relative[0][0]
        result_point = result_relative[0][0]
        
        # Precision should be maintained within reasonable bounds (account for integer conversion)
        precision_threshold = 1e-3
        assert abs(result_point[0] - original_point[0]) < precision_threshold
        assert abs(result_point[1] - original_point[1]) < precision_threshold

    def test_coordinate_bounds_validation(self):
        """Test validation of coordinate bounds."""
        # Test relative coordinates outside [0,1] range
        out_of_bounds_relative = [
            [[-0.1, 0.5], [1.1, 0.5]],  # X coordinates out of bounds
            [[0.5, -0.1], [0.5, 1.1]]   # Y coordinates out of bounds
        ]
        
        # Convert (should still work but produce coordinates outside frame)
        result = relative_to_absolute_coords(
            out_of_bounds_relative,
            self.frame_width,
            self.frame_height
        )
        
        # Verify out-of-bounds coordinates
        assert result[0][0][0] < 0, "Negative relative X should produce negative absolute X"
        assert result[0][1][0] > self.frame_width, "X > 1.0 should produce X > frame_width"
        assert result[1][0][1] < 0, "Negative relative Y should produce negative absolute Y"
        assert result[1][1][1] > self.frame_height, "Y > 1.0 should produce Y > frame_height"

    def test_integer_conversion(self):
        """Test that absolute coordinates are properly converted to integers."""
        # Use coordinates that would produce non-integer results
        fractional_relative = [
            [[0.333333, 0.666666], [0.777777, 0.111111]]
        ]
        
        result = relative_to_absolute_coords(
            fractional_relative,
            self.frame_width,
            self.frame_height
        )
        
        # Verify all coordinates are integers
        for line in result:
            for point in line:
                assert isinstance(point[0], int), "X coordinate should be integer"
                assert isinstance(point[1], int), "Y coordinate should be integer"

    def test_multiple_lines_processing(self):
        """Test processing of multiple lines simultaneously."""
        # Create many lines
        num_lines = 100
        many_lines = []
        
        for i in range(num_lines):
            # Create unique line for each iteration
            x1 = i / (num_lines * 2)
            y1 = i / (num_lines * 2)
            x2 = (i + 1) / (num_lines * 2)
            y2 = (i + 1) / (num_lines * 2)
            many_lines.append([[x1, y1], [x2, y2]])
        
        # Convert all lines
        result = relative_to_absolute_coords(many_lines, self.frame_width, self.frame_height)
        
        # Verify all lines were processed
        assert len(result) == num_lines, "Should process all lines"
        
        # Verify each line has correct structure
        for line in result:
            assert len(line) == 2, "Each line should have 2 points"
            for point in line:
                assert len(point) == 2, "Each point should have 2 coordinates"
                assert isinstance(point[0], int), "X should be integer"
                assert isinstance(point[1], int), "Y should be integer"

    def test_performance_with_large_dataset(self):
        """Test performance with large coordinate datasets."""
        import time
        
        # Create large dataset
        num_lines = 10000
        large_dataset = []
        
        for i in range(num_lines):
            x1 = (i % 100) / 100.0
            y1 = (i % 50) / 50.0
            x2 = ((i + 1) % 100) / 100.0
            y2 = ((i + 1) % 50) / 50.0
            large_dataset.append([[x1, y1], [x2, y2]])
        
        # Measure conversion time
        start_time = time.time()
        result = relative_to_absolute_coords(large_dataset, self.frame_width, self.frame_height)
        conversion_time = time.time() - start_time
        
        # Verify performance and correctness
        assert len(result) == num_lines, "Should process all lines"
        assert conversion_time < 1.0, f"Conversion took too long: {conversion_time:.2f}s"

    # Helper methods
    def _validate_coordinate_format(self, coords):
        """Validate coordinate format."""
        if not isinstance(coords, list):
            return False
        
        for line in coords:
            if not isinstance(line, list) or len(line) != 2:
                return False
            
            for point in line:
                if not isinstance(point, list) or len(point) != 2:
                    return False
                
                if not all(isinstance(coord, (int, float)) for coord in point):
                    return False
        
        return True

    def _calculate_conversion_error(self, original, converted, frame_width, frame_height):
        """Calculate error in coordinate conversion."""
        total_error = 0.0
        num_points = 0
        
        for i, line in enumerate(original):
            for j, point in enumerate(line):
                converted_point = converted[i][j]
                
                # Calculate pixel error
                error_x = abs(point[0] * frame_width - converted_point[0])
                error_y = abs(point[1] * frame_height - converted_point[1])
                
                total_error += error_x + error_y
                num_points += 1
        
        return total_error / num_points if num_points > 0 else 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])