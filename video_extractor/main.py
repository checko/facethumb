#!/usr/bin/env python3
"""
Video Face Extraction Tool - Main Entry Point

Extracts representative JPEG frames from videos containing large faces.
"""

import argparse
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional

# Suppress protobuf deprecation warnings from MediaPipe
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

from video_extractor.utils import (
    setup_logging,
    validate_paths,
    is_video_file,
    get_output_filename,
    DEFAULT_LARGE_FACE_THRESHOLD,
    DEFAULT_START_TIME,
    DEFAULT_GENDER_PREFERENCE,
    DEFAULT_GENDER_WEIGHT,
    DEFAULT_SAMPLING_PERCENTAGES,
    DEFAULT_MAX_SAMPLES
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
from video_extractor.database import VideoDatabase

logger = logging.getLogger("video_extractor")


def process_single_video(
    video_path: Path,
    output_dir: Optional[Path],
    face_analyzer: FaceAnalyzer,
    start_time: float = DEFAULT_START_TIME,
    min_face_threshold: float = DEFAULT_LARGE_FACE_THRESHOLD,
    use_adaptive: bool = True,
    gender_preference: str = DEFAULT_GENDER_PREFERENCE,
    gender_weight: float = DEFAULT_GENDER_WEIGHT
) -> tuple[bool, dict]:
    """
    Process a single video file to extract a face frame using adaptive sampling.

    Adaptive Algorithm (3 stages):
    1. Check embedded thumbnail
    2. Try scene detection timestamps
    3. Adaptive time-based sampling (start_time, then explore)

    Args:
        video_path: Path to video file
        output_dir: Output directory for JPEG (None = use video's parent directory)
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

        import cv2

        best_frame = None
        best_face = None
        best_score = 0.0
        best_timestamp = 0.0

        # Fallback frame for videos with no qualifying faces
        fallback_frame = None
        fallback_timestamp = 0.0

        # Track all detected faces (even below threshold)
        all_faces_detected = []  # List of (frame, face, timestamp) tuples

        # Handle very short videos - extract first frame as fallback
        if duration < 5:
            logger.warning(f"Video too short ({duration:.2f}s), will use first frame")
            try:
                fallback_frame = extract_iframe(video_path, min(1.0, duration * 0.1))
                fallback_timestamp = min(1.0, duration * 0.1)
            except Exception as e:
                logger.debug(f"Failed to extract fallback frame: {e}")

        # Stage 1: Check embedded thumbnail
        if use_adaptive:
            logger.debug("Stage 1: Checking embedded thumbnail")
            thumbnail = extract_thumbnail(video_path)
            if thumbnail is not None:
                # Use thumbnail as fallback if no fallback yet
                if fallback_frame is None:
                    fallback_frame = thumbnail
                    fallback_timestamp = 0

                face_detections = face_analyzer.detect_faces_batch([thumbnail])[0]
                if face_detections:
                    largest_face = find_largest_face(face_detections, thumbnail.shape)

                    # Track this face even if below threshold
                    all_faces_detected.append((thumbnail, largest_face, 0))

                    if largest_face['metrics']['face_area_ratio'] >= min_face_threshold:
                        quality = check_image_quality(thumbnail, largest_face['bbox'])
                        score = score_face(
                            largest_face, thumbnail.shape, thumbnail, quality,
                            gender_preference=gender_preference, gender_weight=gender_weight
                        )

                        logger.debug(f"Thumbnail: face={largest_face['metrics']['face_area_ratio']:.2%}, score={score:.3f}")

                        best_frame = thumbnail
                        best_face = largest_face
                        best_score = score
                        best_timestamp = 0  # Thumbnail

        # Stage 2: Scene detection DISABLED (too slow, often times out)
        # Skipping to Stage 3 for faster, more reliable sampling

        # Stage 3: Dense adaptive time-based sampling
        if best_score < 0.7:  # Only if we haven't found a great face yet
            logger.debug("Stage 3: Dense adaptive sampling")

            # Calculate sampling timestamps (more granular for better coverage)
            timestamps = []

            # Always try the start_time first
            adjusted_start = min(start_time, max(duration * 0.15, 10))
            timestamps.append(adjusted_start)

            # Add dense sampling throughout the video
            if use_adaptive:
                # Use configured sampling percentages (default: every 5% from 5% to 80%)
                # This provides much better coverage for long videos
                for pct in DEFAULT_SAMPLING_PERCENTAGES:
                    ts = duration * pct
                    if 10 < ts < duration - 10 and ts not in timestamps:
                        timestamps.append(ts)

                # Limit to configured max samples for reasonable processing time
                timestamps = timestamps[:DEFAULT_MAX_SAMPLES]

            for ts in timestamps:
                try:
                    frame = extract_iframe(video_path, ts)

                    # Use first successfully extracted frame as fallback if no fallback yet
                    if fallback_frame is None:
                        fallback_frame = frame
                        fallback_timestamp = ts

                    face_detections = face_analyzer.detect_faces_batch([frame])[0]

                    if face_detections:
                        largest_face = find_largest_face(face_detections, frame.shape)

                        # Track this face even if below threshold
                        all_faces_detected.append((frame, largest_face, ts))

                        if largest_face['metrics']['face_area_ratio'] >= min_face_threshold:
                            quality = check_image_quality(frame, largest_face['bbox'])
                            score = score_face(
                                largest_face, frame.shape, frame, quality,
                                gender_preference=gender_preference, gender_weight=gender_weight
                            )

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

        # Fallback strategy: if no qualifying face found
        if best_frame is None or best_face is None:
            # Strategy 1: Use biggest face detected (even if below threshold)
            if all_faces_detected:
                logger.info(f"No faces above threshold, using biggest face found in {video_path.name}")
                # Find the biggest face from all detected faces
                biggest = max(all_faces_detected, key=lambda x: x[1]['metrics']['face_area_ratio'])
                best_frame, best_face, best_timestamp = biggest
                best_score = 0.0  # Indicate this is a fallback
            # Strategy 2: Use first extracted frame (no faces detected at all)
            elif fallback_frame is not None:
                logger.info(f"No faces detected, using first frame from {video_path.name}")
                best_frame = fallback_frame
                best_face = None
                best_timestamp = fallback_timestamp
                best_score = 0.0
            else:
                # This should rarely happen - only if we can't extract any frame
                logger.error(f"Failed to extract any frame from {video_path.name}")
                return False, {'error': 'Failed to extract any frame'}

        # Save best frame
        output_path = get_output_filename(video_path, output_dir)
        cv2.imwrite(str(output_path), best_frame)

        # Get face area ratio if available
        face_area_ratio = best_face['metrics']['face_area_ratio'] if best_face else 0.0

        logger.info(
            f"✓ Extracted frame from {video_path.name} "
            f"(face: {face_area_ratio:.2%}, score: {best_score:.3f}, time: {best_timestamp:.1f}s) -> {output_path.name}"
        )

        # Return metadata for database tracking
        return True, {
            'output_path': output_path,
            'face_area_ratio': face_area_ratio,
            'score': best_score,
            'timestamp_seconds': best_timestamp
        }

    except Exception as e:
        logger.error(f"Failed to process {video_path.name}: {e}")
        return False, {'error': str(e)}


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
    parser.add_argument(
        '--gender-preference', '-g',
        choices=['female', 'male', 'none'],
        default=DEFAULT_GENDER_PREFERENCE,
        help=f'Prefer faces of specific gender (default: {DEFAULT_GENDER_PREFERENCE})'
    )
    parser.add_argument(
        '--gender-weight',
        type=float,
        default=DEFAULT_GENDER_WEIGHT,
        help=f'Weight for gender preference in scoring 0.0-1.0 (default: {DEFAULT_GENDER_WEIGHT})'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume previous processing session (skip already processed videos)'
    )
    parser.add_argument(
        '--db-path',
        default='video_extraction.db',
        help='Path to database file for progress tracking (default: video_extraction.db)'
    )

    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(args.log_level, args.log_file)

    try:
        # Validate paths
        input_path, output_path = validate_paths(args.input, args.output)
        logger.info(f"Input: {input_path}")
        if output_path:
            logger.info(f"Output: {output_path}")
        else:
            logger.info(f"Output: Same folder as each video file")

        # Initialize database if resume enabled or processing directory
        db = None
        if args.resume or input_path.is_dir():
            db = VideoDatabase(args.db_path)
            if args.resume:
                db.reset_processing_videos()  # Clean up interrupted runs
                logger.info("Resume mode enabled")

        # Initialize face analyzer
        face_analyzer = FaceAnalyzer()

        # Process video(s)
        if input_path.is_file():
            # Single video
            success, metadata = process_single_video(
                input_path,
                output_path,
                face_analyzer,
                args.start_time,
                args.threshold,
                use_adaptive=True,
                gender_preference=args.gender_preference,
                gender_weight=args.gender_weight
            )
            sys.exit(0 if success else 1)
        else:
            # Directory of videos
            video_files = [f for f in input_path.rglob('*') if is_video_file(f)]
            logger.info(f"Found {len(video_files)} video files")

            # Filter already processed if resume mode
            if args.resume and db:
                video_files = db.get_unprocessed_videos(video_files)
                logger.info(f"Processing {len(video_files)} unprocessed videos")

            success_count = 0
            for video_file in video_files:
                # Mark as processing
                if db:
                    db.mark_video_processing(video_file)

                # Process video
                success, metadata = process_single_video(
                    video_file,
                    output_path,
                    face_analyzer,
                    args.start_time,
                    args.threshold,
                    use_adaptive=True,
                    gender_preference=args.gender_preference,
                    gender_weight=args.gender_weight
                )

                # Update database
                if db:
                    db.mark_video_processed(
                        video_file,
                        success,
                        output_path=metadata.get('output_path'),
                        face_area_ratio=metadata.get('face_area_ratio'),
                        score=metadata.get('score'),
                        timestamp_seconds=metadata.get('timestamp_seconds'),
                        error_message=metadata.get('error')
                    )

                if success:
                    success_count += 1

            logger.info(f"Successfully processed {success_count}/{len(video_files)} videos")

            # Show stats
            if db:
                stats = db.get_processing_stats()
                logger.info(
                    f"Database stats: {stats['success']} success, "
                    f"{stats['failed']} failed, {stats['total']} total"
                )
                db.close()

            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
