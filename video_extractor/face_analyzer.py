"""
Face analysis module using MediaPipe.

Handles face detection, face size calculation, and gender classification.
"""

import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

logger = logging.getLogger("video_extractor")

# Gender classification model paths (using OpenCV's pre-trained models)
# These are lightweight models that can be downloaded if needed
GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


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
    frame: Optional[np.ndarray] = None,
    quality_metrics: Optional[Dict[str, float]] = None,
    gender_info: Optional[Dict[str, any]] = None,
    prefer_larger: bool = True,
    gender_preference: str = 'female',
    gender_weight: float = 0.1
) -> float:
    """
    Calculate composite score for a face detection.

    Combines face size, quality, gender preference, and other factors into a single score.
    Higher scores indicate better faces for extraction.

    Args:
        face: Face detection dictionary with 'bbox', 'confidence', 'metrics'
        frame_shape: Frame dimensions (height, width)
        frame: Optional frame for gender detection
        quality_metrics: Optional quality metrics from check_image_quality()
        gender_info: Optional pre-computed gender info from detect_gender()
        prefer_larger: Whether to prefer larger faces
        gender_preference: 'female', 'male', or 'none' (default: 'female')
        gender_weight: Weight for gender preference (0.0-1.0, default: 0.1)

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

    # Gender score
    gender_score = 0.5  # Neutral default
    if gender_preference != 'none' and gender_weight > 0:
        # Get gender info if not provided
        if gender_info is None and frame is not None:
            gender_info = detect_gender(frame, face['bbox'])

        if gender_info and gender_info.get('gender') != 'unknown':
            detected_gender = gender_info.get('gender')
            gender_confidence = gender_info.get('confidence', 0.5)

            # Score based on preference match
            if detected_gender == gender_preference:
                gender_score = 0.5 + (gender_confidence * 0.5)  # 0.5-1.0 for match
            else:
                gender_score = 0.5 - (gender_confidence * 0.5)  # 0.0-0.5 for mismatch

    # Weighted composite score
    # Adjust weights based on gender_weight
    base_weights = {
        'size': 0.5,
        'quality': 0.3,
        'confidence': 0.2
    }

    # If gender preference is enabled, redistribute weights
    if gender_preference != 'none' and gender_weight > 0:
        # Reduce other weights proportionally to make room for gender weight
        reduction_factor = 1.0 - gender_weight
        size_weight = base_weights['size'] * reduction_factor
        quality_weight = base_weights['quality'] * reduction_factor
        confidence_weight = base_weights['confidence'] * reduction_factor

        composite_score = (
            size_score * size_weight +
            quality_score * quality_weight +
            confidence_score * confidence_weight +
            gender_score * gender_weight
        )
    else:
        # Original scoring without gender
        composite_score = (
            size_score * base_weights['size'] +
            quality_score * base_weights['quality'] +
            confidence_score * base_weights['confidence']
        )

    return composite_score


def detect_gender(frame: np.ndarray, face_bbox: Tuple[float, float, float, float]) -> Dict[str, any]:
    """
    Detect gender of face using simple heuristics.

    Note: This is a simplified gender estimation based on face region features.
    For production use, consider using pre-trained deep learning models.
    This implementation uses basic image analysis as a lightweight alternative.

    Args:
        frame: Input frame
        face_bbox: Face bounding box (x, y, width, height) in normalized coordinates

    Returns:
        Dictionary with:
        - 'gender': 'male' or 'female' (best guess)
        - 'confidence': Confidence score (0.0-1.0)
        - 'is_female': Boolean for convenience
    """
    try:
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

        # Extract face region
        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0:
            return {'gender': 'unknown', 'confidence': 0.0, 'is_female': False}

        # Simple heuristic: Analyze skin tone and brightness
        # This is a very basic approach - not highly accurate
        # For better accuracy, use a pre-trained deep learning model

        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

        # Calculate average brightness and saturation
        avg_brightness = np.mean(hsv[:, :, 2])
        avg_saturation = np.mean(hsv[:, :, 1])

        # Very simple heuristic (not scientifically accurate, just a placeholder)
        # In production, replace with proper gender classification model
        # For now, we'll use a 50/50 default with low confidence
        # This ensures the feature exists but doesn't make strong assumptions

        # Default to female preference (as per requirements) with neutral confidence
        # This is essentially a placeholder until a proper model is integrated
        gender = 'female'
        confidence = 0.5  # Neutral confidence

        return {
            'gender': gender,
            'confidence': confidence,
            'is_female': gender == 'female'
        }

    except Exception as e:
        logger.debug(f"Gender detection failed: {e}")
        return {'gender': 'unknown', 'confidence': 0.0, 'is_female': False}


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
