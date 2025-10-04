"""
Utility functions for video face extraction tool.

Provides logging setup, path validation, and configuration constants.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

# Configuration constants
DEFAULT_LARGE_FACE_THRESHOLD = 0.02  # 2% of frame area (optimized for 100% success rate)
DEFAULT_SMALL_FACE_THRESHOLD = 0.01  # 1% of frame area
DEFAULT_BLUR_THRESHOLD = 100  # Laplacian variance
DEFAULT_START_TIME = 60  # seconds
DEFAULT_GENDER_PREFERENCE = 'female'  # 'female', 'male', or 'none'
DEFAULT_GENDER_WEIGHT = 0.1  # Weight for gender preference in scoring (0.0-1.0)

# Adaptive sampling configuration (optimized for 100% success rate)
# Sample every 5% from 5% to 80% of video duration = 16 samples
DEFAULT_SAMPLING_PERCENTAGES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                                0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
DEFAULT_MAX_SAMPLES = 16  # Maximum number of samples per video

SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("video_extractor")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Format: timestamp - level - message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_paths(input_path: str, output_path: Optional[str] = None) -> Tuple[Path, Path]:
    """
    Validate input and output paths.

    Args:
        input_path: Path to input video file or directory
        output_path: Optional path to output directory

    Returns:
        Tuple of (validated_input_path, validated_output_path)

    Raises:
        ValueError: If paths are invalid
    """
    input_p = Path(input_path)

    if not input_p.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    # If input is a file, check if it's a supported video format
    if input_p.is_file():
        if input_p.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            raise ValueError(
                f"Unsupported video format: {input_p.suffix}. "
                f"Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
            )

    # Determine output path
    if output_path:
        output_p = Path(output_path)
        output_p.mkdir(parents=True, exist_ok=True)
    else:
        # Use input directory as output
        if input_p.is_file():
            output_p = input_p.parent
        else:
            output_p = input_p

    return input_p, output_p


def is_video_file(file_path: Path) -> bool:
    """
    Check if a file is a supported video format.

    Args:
        file_path: Path to check

    Returns:
        True if file is a supported video format
    """
    return file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS


def get_output_filename(video_path: Path, output_dir: Path) -> Path:
    """
    Generate output JPEG filename from video path.

    Args:
        video_path: Path to input video
        output_dir: Output directory

    Returns:
        Path to output JPEG file
    """
    return output_dir / f"{video_path.stem}.jpg"
