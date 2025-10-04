#!/usr/bin/env python3
"""
Video Face Extraction Tool - Main Entry Point

Extracts representative JPEG frames from videos containing large faces.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from video_extractor.utils import (
    setup_logging,
    validate_paths,
    is_video_file,
    get_output_filename,
    DEFAULT_LARGE_FACE_THRESHOLD,
    DEFAULT_START_TIME
)
from video_extractor.video_processor import (
    get_video_metadata,
    extract_iframe
)
from video_extractor.face_analyzer import (
    FaceAnalyzer,
    find_largest_face
)

logger = logging.getLogger("video_extractor")


def process_single_video(
    video_path: Path,
    output_dir: Path,
    face_analyzer: FaceAnalyzer,
    start_time: float = DEFAULT_START_TIME,
    min_face_threshold: float = DEFAULT_LARGE_FACE_THRESHOLD
) -> bool:
    """
    Process a single video file to extract a face frame.

    Args:
        video_path: Path to video file
        output_dir: Output directory for JPEG
        face_analyzer: Face analyzer instance
        start_time: Timestamp to extract frame (seconds)
        min_face_threshold: Minimum face area ratio

    Returns:
        True if frame extracted successfully, False otherwise
    """
    logger.info(f"Processing: {video_path.name}")

    try:
        # Get video metadata
        metadata = get_video_metadata(video_path)
        duration = metadata['duration']
        logger.debug(f"Video duration: {duration:.2f}s, Resolution: {metadata['width']}x{metadata['height']}")

        # Adjust start time if video is too short
        actual_start_time = min(start_time, max(duration * 0.15, 10))
        if actual_start_time >= duration - 5:
            logger.warning(f"Video too short ({duration:.2f}s), skipping")
            return False

        # Extract frame at timestamp
        logger.debug(f"Extracting frame at {actual_start_time:.2f}s")
        frame = extract_iframe(video_path, actual_start_time)

        # Detect faces
        face_detections = face_analyzer.detect_faces_batch([frame])[0]

        if not face_detections:
            logger.info(f"No faces detected in {video_path.name}")
            return False

        # Find largest face
        largest_face = find_largest_face(face_detections, frame.shape)
        face_area_ratio = largest_face['metrics']['face_area_ratio']

        logger.debug(f"Largest face area ratio: {face_area_ratio:.4f}")

        # Check if face meets threshold
        if face_area_ratio < min_face_threshold:
            logger.info(
                f"Face too small ({face_area_ratio:.2%} < {min_face_threshold:.2%}) in {video_path.name}"
            )
            return False

        # Save frame
        output_path = get_output_filename(video_path, output_dir)
        import cv2
        cv2.imwrite(str(output_path), frame)

        logger.info(
            f"✓ Extracted frame from {video_path.name} "
            f"(face: {face_area_ratio:.2%}, time: {actual_start_time:.1f}s) -> {output_path.name}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to process {video_path.name}: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract representative JPEG frames from videos with faces"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input video file or directory'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output directory (default: same as input)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=DEFAULT_LARGE_FACE_THRESHOLD,
        help=f'Minimum face area ratio (default: {DEFAULT_LARGE_FACE_THRESHOLD})'
    )
    parser.add_argument(
        '--start-time', '-s',
        type=float,
        default=DEFAULT_START_TIME,
        help=f'Initial timestamp to check in seconds (default: {DEFAULT_START_TIME})'
    )
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        help='Write logs to file'
    )

    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(args.log_level, args.log_file)

    try:
        # Validate paths
        input_path, output_path = validate_paths(args.input, args.output)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")

        # Initialize face analyzer
        face_analyzer = FaceAnalyzer()

        # Process video(s)
        if input_path.is_file():
            # Single video
            success = process_single_video(
                input_path,
                output_path,
                face_analyzer,
                args.start_time,
                args.threshold
            )
            sys.exit(0 if success else 1)
        else:
            # Directory of videos
            video_files = [f for f in input_path.rglob('*') if is_video_file(f)]
            logger.info(f"Found {len(video_files)} video files")

            success_count = 0
            for video_file in video_files:
                if process_single_video(
                    video_file,
                    output_path,
                    face_analyzer,
                    args.start_time,
                    args.threshold
                ):
                    success_count += 1

            logger.info(f"Successfully processed {success_count}/{len(video_files)} videos")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
