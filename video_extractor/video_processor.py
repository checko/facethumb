"""
Video processing module for frame extraction.

Handles video I/O operations using FFmpeg including metadata retrieval
and frame extraction.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import cv2

logger = logging.getLogger("video_extractor")


def get_video_metadata(video_path: Path) -> Dict[str, any]:
    """
    Get video metadata including duration and resolution.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with 'duration', 'width', 'height', 'fps'

    Raises:
        RuntimeError: If FFprobe fails
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break

        if not video_stream:
            raise RuntimeError(f"No video stream found in {video_path}")

        # Extract metadata
        duration = float(data.get('format', {}).get('duration', 0))
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))

        # Calculate FPS
        fps_str = video_stream.get('r_frame_rate', '0/1')
        if '/' in fps_str:
            num, denom = fps_str.split('/')
            fps = float(num) / float(denom) if float(denom) != 0 else 0
        else:
            fps = float(fps_str)

        return {
            'duration': duration,
            'width': width,
            'height': height,
            'fps': fps
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe failed for {video_path}: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Failed to get metadata for {video_path}: {str(e)}")


def extract_iframe(video_path: Path, timestamp: float, output_path: Optional[Path] = None) -> np.ndarray:
    """
    Extract a single I-frame at the specified timestamp.

    Args:
        video_path: Path to video file
        timestamp: Time in seconds to extract frame
        output_path: Optional path to save frame as image

    Returns:
        Frame as numpy array (BGR format)

    Raises:
        RuntimeError: If frame extraction fails
    """
    try:
        # Use FFmpeg to extract frame at timestamp
        # -ss: seek to timestamp
        # -i: input file
        # -vframes 1: extract 1 frame
        # -f image2pipe: output to pipe
        # -vcodec png: use PNG for lossless extraction
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),
            '-i', str(video_path),
            '-vframes', '1',
            '-f', 'image2pipe',
            '-vcodec', 'png',
            '-'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True
        )

        # Decode image from bytes
        img_array = np.frombuffer(result.stdout, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise RuntimeError(f"Failed to decode frame at {timestamp}s")

        # Optionally save to file
        if output_path:
            cv2.imwrite(str(output_path), frame)
            logger.debug(f"Frame saved to {output_path}")

        return frame

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to extract frame: {e.stderr.decode()}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract frame at {timestamp}s: {str(e)}")


def extract_thumbnail(video_path: Path) -> Optional[np.ndarray]:
    """
    Extract embedded thumbnail from video file (if available).

    Many video files have embedded thumbnails that can be quickly extracted
    without decoding the video stream.

    Args:
        video_path: Path to video file

    Returns:
        Thumbnail as numpy array (BGR format), or None if no thumbnail
    """
    try:
        # Try to extract embedded thumbnail using FFmpeg
        # -map 0:v:1 selects the second video stream (often the thumbnail)
        # Some videos store thumbnails in attached_pic disposition
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-map', '0:v',
            '-vframes', '1',
            '-f', 'image2pipe',
            '-vcodec', 'png',
            '-'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=5,
            stderr=subprocess.DEVNULL  # Suppress FFmpeg logs
        )

        if result.returncode == 0 and result.stdout:
            # Decode image from bytes
            img_array = np.frombuffer(result.stdout, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is not None:
                logger.debug(f"Extracted embedded thumbnail from {video_path.name}")
                return frame

        return None

    except subprocess.TimeoutExpired:
        logger.debug(f"Thumbnail extraction timeout for {video_path.name}")
        return None
    except Exception as e:
        logger.debug(f"No thumbnail found in {video_path.name}: {e}")
        return None


def detect_scenes(video_path: Path, threshold: float = 0.3, max_scenes: int = 10) -> list:
    """
    Detect scene changes in video using FFmpeg scene detection.

    Returns timestamps of significant scene changes where interesting
    content (like faces) might appear.

    Args:
        video_path: Path to video file
        threshold: Scene change threshold (0.0-1.0, lower = more sensitive)
        max_scenes: Maximum number of scene timestamps to return

    Returns:
        List of timestamps (in seconds) where scenes change
    """
    try:
        # Use FFmpeg's scene detection filter
        # select='gt(scene,threshold)' detects scene changes
        # showinfo outputs timestamp info
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'select=gt(scene\\,{threshold}),showinfo',
            '-f', 'null',
            '-'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # Limit to 30 seconds for scene detection
        )

        # Parse timestamps from showinfo output
        # Look for patterns like "pts_time:123.456"
        timestamps = []
        for line in result.stderr.split('\n'):
            if 'pts_time:' in line:
                try:
                    # Extract timestamp
                    time_str = line.split('pts_time:')[1].split()[0]
                    timestamp = float(time_str)
                    timestamps.append(timestamp)

                    if len(timestamps) >= max_scenes:
                        break
                except (ValueError, IndexError):
                    continue

        if timestamps:
            logger.debug(f"Found {len(timestamps)} scene changes in {video_path.name}")
            return timestamps[:max_scenes]

        return []

    except subprocess.TimeoutExpired:
        logger.warning(f"Scene detection timeout for {video_path.name}")
        return []
    except Exception as e:
        logger.debug(f"Scene detection failed for {video_path.name}: {e}")
        return []


def extract_frame_cluster(video_path: Path, start_time: float, count: int = 3, interval: float = 0.1) -> list:
    """
    Extract a cluster of frames around a timestamp.

    Args:
        video_path: Path to video file
        start_time: Starting timestamp in seconds
        count: Number of frames to extract
        interval: Time interval between frames in seconds

    Returns:
        List of frames as numpy arrays
    """
    frames = []
    for i in range(count):
        timestamp = start_time + (i * interval)
        try:
            frame = extract_iframe(video_path, timestamp)
            frames.append(frame)
        except Exception as e:
            logger.warning(f"Failed to extract frame at {timestamp}s: {e}")

    return frames
