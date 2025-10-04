# Project Requirements: Video Face Extraction Tool

## Overview
A command-line tool to process large collections of video files, extracting a single representative JPEG frame per video that contains a large, identifiable human face (preferably female). The tool uses adaptive sampling for efficiency, focusing on finding faces without decoding entire videos. Designed for Linux environments with batch processing capabilities.

## Functional Requirements
1. **Input Handling**:
   - Accept a directory path containing video files (e.g., MP4, AVI, MOV).
   - Support recursive scanning of subdirectories.
   - Validate inputs: Skip unsupported formats or corrupted files with logging.

2. **Frame Extraction**:
   - Use adaptive algorithm: Start at 1 minute, extract 2-3 frame clusters, skip based on face presence/size.
   - Prioritize I-frames or nearest frames for speed.
   - Extract up to 10-15 frames per video max to avoid excessive processing.

3. **Face Detection and Analysis**:
   - Detect faces using deep learning (OpenCV DNN with SSD model).
   - Calculate face size (bounding box area relative to frame).
   - Classify gender (male/female) with a pre-trained model; prefer female faces.
   - Criteria for "qualifying frame": Face >10% frame area, preferred gender.

4. **Output**:
   - Save one JPEG per video (filename: video.mp4 → video.jpg).
   - Place JPEGs in input directory or specified output directory.
   - Log results: Success (with timestamp/frame details), failures, skipped videos.

5. **Processing Modes**:
   - Batch mode: Process entire directory.
   - Parallel processing: Use multiple CPU cores for speed.

## Non-Functional Requirements
1. **Performance**:
   - Process 10-50 videos per minute on a standard Linux machine (depending on video length).
   - Minimize CPU/GPU usage; support GPU acceleration if available.

2. **Accuracy**:
   - Face detection: >90% accuracy on varied conditions (lighting, angles).
   - Gender classification: >85% accuracy.
   - False positive rate: Low (avoid saving frames without qualifying faces).

3. **Usability**:
   - Command-line interface: Simple args (e.g., --input, --output, --thresholds).
   - Error handling: Graceful failures with clear logs (no crashes).
   - Documentation: Inline comments, README for setup.

4. **Reliability**:
   - Handle edge cases: Short videos (<1min), no faces, multiple faces.
   - Robust to video corruption or missing dependencies.

## Constraints and Assumptions
- **Environment**: Linux (Ubuntu/Debian), Python 3.8+, FFmpeg installed.
- **Dependencies**: OpenCV, FFmpeg-Python (install via pip).
- **Video Formats**: Common codecs (H.264, etc.); assume standard resolutions (720p-4K).
- **Assumptions**: Videos are user-generated (not encrypted); faces are human (not animals/masks).
- **Limitations**: No real-time processing; offline batch only.

## Success Criteria
- Tool runs without errors on test set of 100 videos.
- Extracts JPEGs for 80%+ of videos with faces.
- Processing time: <5 minutes for 100 videos.

## Out of Scope
- Video editing/modification.
- Multiple outputs per video.
- Cloud integration or GUI.