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
    extract_iframe,
    extract_thumbnail,
    detect_scenes
)
from video_extractor.face_analyzer import (
    FaceAnalyzer,
    find_largest_face,
    check_image_quality,
    score_face
)

logger = logging.getLogger("video_extractor")


def process_single_video(
    video_path: Path,
    output_dir: Path,
    face_analyzer: FaceAnalyzer,
    start_time: float = DEFAULT_START_TIME,
    min_face_threshold: float = DEFAULT_LARGE_FACE_THRESHOLD,
    use_adaptive: bool = True
) -> bool:
    """
    Process a single video file to extract a face frame using adaptive sampling.

    Adaptive Algorithm (3 stages):
    1. Check embedded thumbnail
    2. Try scene detection timestamps
    3. Adaptive time-based sampling (start_time, then explore)

    Args:
        video_path: Path to video file
        output_dir: Output directory for JPEG
        face_analyzer: Face analyzer instance
        start_time: Initial timestamp to check (seconds)
        min_face_threshold: Minimum face area ratio
        use_adaptive: Use adaptive algorithm (default: True)

    Returns:
        True if frame extracted successfully, False otherwise
    """
    logger.info(f"Processing: {video_path.name}")

    try:
        # Get video metadata
        metadata = get_video_metadata(video_path)
        duration = metadata['duration']
        logger.debug(f"Video duration: {duration:.2f}s, Resolution: {metadata['width']}x{metadata['height']}")

        if duration < 5:
            logger.warning(f"Video too short ({duration:.2f}s), skipping")
            return False

        import cv2

        best_frame = None
        best_face = None
        best_score = 0.0
        best_timestamp = 0.0

        # Stage 1: Check embedded thumbnail
        if use_adaptive:
            logger.debug("Stage 1: Checking embedded thumbnail")
            thumbnail = extract_thumbnail(video_path)
            if thumbnail is not None:
                face_detections = face_analyzer.detect_faces_batch([thumbnail])[0]
                if face_detections:
                    largest_face = find_largest_face(face_detections, thumbnail.shape)
                    if largest_face['metrics']['face_area_ratio'] >= min_face_threshold:
                        quality = check_image_quality(thumbnail, largest_face['bbox'])
                        score = score_face(largest_face, thumbnail.shape, quality)

                        logger.debug(f"Thumbnail: face={largest_face['metrics']['face_area_ratio']:.2%}, score={score:.3f}")

                        best_frame = thumbnail
                        best_face = largest_face
                        best_score = score
                        best_timestamp = 0  # Thumbnail

        # Stage 2: Try scene detection timestamps
        if use_adaptive and best_score < 0.6:  # Only if thumbnail wasn't great
            logger.debug("Stage 2: Scene detection")
            scene_timestamps = detect_scenes(video_path, threshold=0.4, max_scenes=5)

            for ts in scene_timestamps[:3]:  # Check first 3 scenes
                if ts < 5 or ts > duration - 5:
                    continue

                try:
                    frame = extract_iframe(video_path, ts)
                    face_detections = face_analyzer.detect_faces_batch([frame])[0]

                    if face_detections:
                        largest_face = find_largest_face(face_detections, frame.shape)
                        if largest_face['metrics']['face_area_ratio'] >= min_face_threshold:
                            quality = check_image_quality(frame, largest_face['bbox'])
                            score = score_face(largest_face, frame.shape, quality)

                            logger.debug(f"Scene @{ts:.1f}s: face={largest_face['metrics']['face_area_ratio']:.2%}, score={score:.3f}")

                            if score > best_score:
                                best_frame = frame
                                best_face = largest_face
                                best_score = score
                                best_timestamp = ts

                            # If we found a great face, stop early
                            if score > 0.8:
                                break
                except Exception as e:
                    logger.debug(f"Failed to extract scene at {ts:.1f}s: {e}")

        # Stage 3: Adaptive time-based sampling
        if best_score < 0.7:  # Only if we haven't found a great face yet
            logger.debug("Stage 3: Adaptive sampling")

            # Calculate sampling timestamps (spread across video)
            timestamps = []

            # Always try the start_time
            adjusted_start = min(start_time, max(duration * 0.15, 10))
            timestamps.append(adjusted_start)

            # Add more timestamps if needed
            if use_adaptive:
                # Sample at 15%, 30%, 45%, 60% of video duration
                for pct in [0.15, 0.30, 0.45, 0.60]:
                    ts = duration * pct
                    if 10 < ts < duration - 10 and ts not in timestamps:
                        timestamps.append(ts)

                # Limit to 5 total samples
                timestamps = timestamps[:5]

            for ts in timestamps:
                try:
                    frame = extract_iframe(video_path, ts)
                    face_detections = face_analyzer.detect_faces_batch([frame])[0]

                    if face_detections:
                        largest_face = find_largest_face(face_detections, frame.shape)
                        if largest_face['metrics']['face_area_ratio'] >= min_face_threshold:
                            quality = check_image_quality(frame, largest_face['bbox'])
                            score = score_face(largest_face, frame.shape, quality)

                            logger.debug(f"Sample @{ts:.1f}s: face={largest_face['metrics']['face_area_ratio']:.2%}, score={score:.3f}")

                            if score > best_score:
                                best_frame = frame
                                best_face = largest_face
                                best_score = score
                                best_timestamp = ts

                            # If we found a great face, stop early
                            if score > 0.8:
                                break
                except Exception as e:
                    logger.debug(f"Failed to extract frame at {ts:.1f}s: {e}")

        # Check if we found a qualifying face
        if best_frame is None or best_face is None:
            logger.info(f"No qualifying faces found in {video_path.name}")
            return False

        # Save best frame
        output_path = get_output_filename(video_path, output_dir)
        cv2.imwrite(str(output_path), best_frame)

        face_area_ratio = best_face['metrics']['face_area_ratio']
        logger.info(
            f"✓ Extracted frame from {video_path.name} "
            f"(face: {face_area_ratio:.2%}, score: {best_score:.3f}, time: {best_timestamp:.1f}s) -> {output_path.name}"
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
