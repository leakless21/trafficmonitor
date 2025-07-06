"""
Utility functions for the traffic monitor system.
"""

from typing import List


def relative_to_absolute_coords(relative_lines: List[List[List[float]]], frame_width: int, frame_height: int) -> List[List[List[int]]]:
    """
    Convert relative coordinates (0.0-1.0) to absolute pixel coordinates for multiple lines.
    
    Args:
        relative_lines: List of lines, where each line contains coordinate pairs [[x1, y1], [x2, y2]] in relative format (0.0-1.0)
        frame_width: Width of the frame in pixels
        frame_height: Height of the frame in pixels
    
    Returns:
        List of lines with coordinate pairs in absolute pixel format [[[x1, y1], [x2, y2]], ...]
    """
    absolute_lines = []
    for line in relative_lines:
        absolute_line = []
        for coord_pair in line:
            absolute_pair = [
                int(coord_pair[0] * frame_width),   # x coordinate
                int(coord_pair[1] * frame_height)   # y coordinate
            ]
            absolute_line.append(absolute_pair)
        absolute_lines.append(absolute_line)
    return absolute_lines


def absolute_to_relative_coords(absolute_lines: List[List[List[int]]], frame_width: int, frame_height: int) -> List[List[List[float]]]:
    """
    Convert absolute pixel coordinates to relative coordinates (0.0-1.0) for multiple lines.
    
    Args:
        absolute_lines: List of lines, where each line contains coordinate pairs [[x1, y1], [x2, y2]] in absolute pixel format
        frame_width: Width of the frame in pixels
        frame_height: Height of the frame in pixels
    
    Returns:
        List of lines with coordinate pairs in relative format (0.0-1.0) [[[x1, y1], [x2, y2]], ...]
    """
    relative_lines = []
    for line in absolute_lines:
        relative_line = []
        for coord_pair in line:
            relative_pair = [
                coord_pair[0] / frame_width,    # x coordinate
                coord_pair[1] / frame_height    # y coordinate
            ]
            relative_line.append(relative_pair)
        relative_lines.append(relative_line)
    return relative_lines 