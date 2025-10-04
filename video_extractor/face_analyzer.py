"""
Face analysis module using MediaPipe.

Handles face detection and face size calculation.
"""

import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import mediapipe as mp

logger = logging.getLogger("video_extractor")


class FaceAnalyzer:
    """Face analyzer using MediaPipe Face Detection."""

    def __init__(self, min_detection_confidence: float = 0.7):
        """
        Initialize face analyzer.

        Args:
            min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=1  # 1 = full range model (better accuracy), 0 = short range
        )
        logger.info(f"Face analyzer initialized with MediaPipe (confidence: {min_detection_confidence})")

    def detect_faces_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect faces in a batch of frames.

        Args:
            frames: List of frames as numpy arrays (BGR format)

        Returns:
            List of face detections for each frame. Each detection is a dict with:
            - 'bbox': (x, y, width, height) normalized coordinates (0.0-1.0)
            - 'confidence': Detection confidence score
        """
        results = []

        for frame in frames:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            detection_result = self.face_detection.process(rgb_frame)

            frame_faces = []
            if detection_result.detections:
                for detection in detection_result.detections:
                    # Get bounding box (normalized coordinates)
                    bbox = detection.location_data.relative_bounding_box

                    # Validate face has proper landmarks (eyes, nose, mouth, etc.)
                    # MediaPipe provides 6 key points: right_eye, left_eye, nose_tip,
                    # mouth_center, right_ear, left_ear
                    keypoints = detection.location_data.relative_keypoints

                    # Check if we have at least eyes and nose (minimum for valid face)
                    has_valid_landmarks = len(keypoints) >= 3

                    # Additional validation: check aspect ratio (faces shouldn't be too wide/tall)
                    aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 0
                    is_valid_aspect_ratio = 0.5 < aspect_ratio < 2.0  # Reasonable face proportions

                    # Only add if it passes validation
                    if has_valid_landmarks and is_valid_aspect_ratio:
                        face_info = {
                            'bbox': (bbox.xmin, bbox.ymin, bbox.width, bbox.height),
                            'confidence': detection.score[0],
                            'keypoints': len(keypoints)
                        }
                        frame_faces.append(face_info)
                    else:
                        logger.debug(
                            f"Rejected face: landmarks={len(keypoints)}, "
                            f"aspect_ratio={aspect_ratio:.2f}"
                        )

            results.append(frame_faces)

        return results

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


def calculate_face_metrics(face_bbox: Tuple[float, float, float, float],
                          frame_shape: Tuple[int, int]) -> Dict[str, float]:
    """
    Calculate face metrics including size relative to frame.

    Args:
        face_bbox: Face bounding box (x, y, width, height) in normalized coordinates (0.0-1.0)
        frame_shape: Frame dimensions (height, width)

    Returns:
        Dictionary with:
        - 'face_area_ratio': Face area as ratio of frame area (0.0-1.0)
        - 'face_width_pixels': Face width in pixels
        - 'face_height_pixels': Face height in pixels
    """
    x, y, w, h = face_bbox
    frame_height, frame_width = frame_shape[:2]

    # Calculate face area ratio (face area / frame area)
    face_area_ratio = w * h  # Already normalized, so this is the ratio

    # Calculate pixel dimensions
    face_width_pixels = w * frame_width
    face_height_pixels = h * frame_height

    return {
        'face_area_ratio': face_area_ratio,
        'face_width_pixels': face_width_pixels,
        'face_height_pixels': face_height_pixels
    }


def find_largest_face(faces: List[Dict], frame_shape: Tuple[int, int]) -> Optional[Dict]:
    """
    Find the largest face in a list of detections.

    Args:
        faces: List of face detections
        frame_shape: Frame dimensions (height, width)

    Returns:
        Face with largest area, or None if no faces
    """
    if not faces:
        return None

    largest_face = None
    largest_area = 0.0

    for face in faces:
        metrics = calculate_face_metrics(face['bbox'], frame_shape)
        if metrics['face_area_ratio'] > largest_area:
            largest_area = metrics['face_area_ratio']
            largest_face = {
                **face,
                'metrics': metrics
            }

    return largest_face


def check_image_quality(frame: np.ndarray, face_bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, float]:
    """
    Check image quality using blur detection.

    Uses Laplacian variance to detect blur. Higher values indicate sharper images.

    Args:
        frame: Input frame
        face_bbox: Optional face bounding box to check quality of face region only

    Returns:
        Dictionary with:
        - 'blur_score': Laplacian variance (higher = sharper, typically >100 is good)
        - 'is_blurry': Boolean indicating if image is too blurry
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If face bbox provided, crop to face region
    if face_bbox:
        x, y, w, h = face_bbox
        frame_height, frame_width = frame.shape[:2]

        # Convert normalized coordinates to pixels
        x1 = int(x * frame_width)
        y1 = int(y * frame_height)
        x2 = int((x + w) * frame_width)
        y2 = int((y + h) * frame_height)

        # Ensure coordinates are within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)

        # Crop to face region
        gray = gray[y1:y2, x1:x2]

    # Calculate Laplacian variance (blur metric)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # Threshold for blur detection (can be adjusted)
    blur_threshold = 100.0
    is_blurry = blur_score < blur_threshold

    return {
        'blur_score': blur_score,
        'is_blurry': is_blurry
    }


def score_face(
    face: Dict,
    frame_shape: Tuple[int, int],
    quality_metrics: Optional[Dict[str, float]] = None,
    prefer_larger: bool = True
) -> float:
    """
    Calculate composite score for a face detection.

    Combines face size, quality, and other factors into a single score.
    Higher scores indicate better faces for extraction.

    Args:
        face: Face detection dictionary with 'bbox', 'confidence', 'metrics'
        frame_shape: Frame dimensions (height, width)
        quality_metrics: Optional quality metrics from check_image_quality()
        prefer_larger: Whether to prefer larger faces

    Returns:
        Composite score (0.0-1.0+, higher is better)
    """
    # Base score from face area ratio
    if 'metrics' not in face:
        face['metrics'] = calculate_face_metrics(face['bbox'], frame_shape)

    face_area_ratio = face['metrics']['face_area_ratio']

    # Size score (0.0-1.0, normalized by expected max of 0.5 = 50% of frame)
    size_score = min(face_area_ratio / 0.5, 1.0) if prefer_larger else face_area_ratio

    # Confidence score (already 0.0-1.0)
    confidence_score = face.get('confidence', 0.5)

    # Quality score
    quality_score = 1.0
    if quality_metrics:
        # Normalize blur score (typically 0-500, with >100 being good)
        blur_score = quality_metrics.get('blur_score', 100)
        quality_score = min(blur_score / 200.0, 1.0)  # Normalize to 0-1

    # Weighted composite score
    # Size is most important (50%), then quality (30%), then confidence (20%)
    composite_score = (
        size_score * 0.5 +
        quality_score * 0.3 +
        confidence_score * 0.2
    )

    return composite_score


def draw_face_box(frame: np.ndarray, face_bbox: Tuple[float, float, float, float],
                  color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box around detected face.

    Args:
        frame: Input frame
        face_bbox: Face bounding box (x, y, width, height) in normalized coordinates
        color: Box color in BGR format
        thickness: Line thickness

    Returns:
        Frame with drawn bounding box
    """
    x, y, w, h = face_bbox
    frame_height, frame_width = frame.shape[:2]

    # Convert normalized coordinates to pixels
    x1 = int(x * frame_width)
    y1 = int(y * frame_height)
    x2 = int((x + w) * frame_width)
    y2 = int((y + h) * frame_height)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    return frame
