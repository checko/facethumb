# Video Face Extraction Algorithm: Adaptive Frame Sampling with Smart Optimization

## Overview
This algorithm efficiently locates a frame containing a large, identifiable face (preferably female) within a video file. It uses multi-stage optimization: embedded thumbnail extraction, scene detection, I-frame sampling, and adaptive skipping to minimize processing time. Designed for batch processing large video collections on Linux laptops.

## Key Concepts
- **Face Criteria**: A qualifying face must meet resolution-adaptive size threshold (typically >10% of frame area for 1080p), pass quality checks (not blurred/occluded), and preferably be female.
- **Efficiency Heuristics**:
  - Embedded thumbnails first → Instant results if suitable.
  - Scene detection → Focus on visually distinct sections.
  - I-frames only → Avoid decoding overhead.
  - No faces early → Likely "no-person" video; skip aggressively.
  - Small faces → Person present; check nearby for zooms.
- **Tools**: FFmpeg for frame extraction/scene detection, MediaPipe for face detection, pre-trained model for gender classification.

## Algorithm Steps

### Stage 1: Quick Checks (0-2 seconds per video)
1. **Embedded Thumbnail Extraction**:
   - Extract video's embedded thumbnail using FFmpeg (`-map 0:v:1` or metadata).
   - If thumbnail exists: Run face detection → If qualifying face found, save and exit.

2. **Scene Detection**:
   - Run FFmpeg scene detection (`-vf select='gt(scene,0.3)'`) to identify keyframes.
   - Limit to top 15-20 scene changes for efficiency.

### Stage 2: Initialization
3. **Dynamic Configuration**:
   - Get video duration and resolution.
   - `start_time = max(video_duration * 0.15, 20)` seconds (15% into video or 20s minimum).
   - Calculate adaptive thresholds:
     - `large_face_threshold = 0.10 * (1080 / frame_height)^2` (scale for resolution)
     - `small_face_threshold = 0.02 * (1080 / frame_height)^2`
     - `blur_threshold = 100` (Laplacian variance for sharpness)
   - Counters: `consecutive_no_faces = 0`, `frames_processed = 0`.

### Stage 3: Main Loop (Repeat Until Match or Video End)
4. **I-Frame Extraction**:
   - At `start_time`, extract 1 I-frame using FFmpeg (`-skip_frame nokey`).
   - Fallback to scene-detected keyframes if available.
   - **Quality Check**: Calculate Laplacian variance → Skip if blurred.

5. **Batch Face Detection** (every 3-5 frames):
   - Collect frames into batch for GPU-accelerated processing.
   - Use MediaPipe Face Detection for face/landmark detection.
   - For detected faces: Calculate bounding box size, check occlusion (via landmarks).

6. **Gender Classification**:
   - For faces ≥ `small_face_threshold`: Run gender classifier.
   - Score faces: `score = face_size * gender_preference_weight * quality_score`.

7. **Selection Logic**:
   - If any face ≥ `large_face_threshold`, preferred gender, and quality > threshold:
     - Save frame as JPEG (pick highest scoring if multiple) → Exit.

8. **Adaptive Skipping**:
   - **No Faces in Frame**:
     - `consecutive_no_faces += 1`
     - If `consecutive_no_faces >= 3`: Jump +5 minutes (`start_time += 300`), reset counter.
     - Else: Jump to next scene change or +1 minute (`start_time += 60`).
   - **Small Faces Present**: Jump +30 seconds or to next scene (`start_time += 30`), reset counter.
   - **Frame Limit**: If `frames_processed >= 15`: Stop, use best fallback.

9. **Termination**:
   - If `start_time > video_duration - 30`: Stop.
   - Fallback: Save highest-scoring frame found (if any), otherwise skip video.

## Parameters and Tuning
- **Scene Detection Threshold**: `0.3` (adjust for video types: lower for slow videos, higher for fast cuts).
- **Batch Size**: 3-5 frames for GPU efficiency.
- **Blur Threshold**: 100 (Laplacian variance; higher = sharper required).
- **Gender Preference Weight**: 1.5x for female faces (configurable).
- Expected: 5-10 frames/video, <3 seconds processing time per video.

## Edge Cases
- **Short videos (<2min)**: Skip to scene keyframes only, no time-based jumps.
- **No embedded thumbnail**: Continue to scene detection.
- **No scenes detected**: Fall back to time-based sampling.
- **No qualifying match**: Save best available frame or skip entirely (configurable).
- **Multiple faces**: Select highest composite score (size × gender × quality).