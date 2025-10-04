# System Design: Video Face Extraction Tool

## Overview
A Python-based tool for extracting representative JPEG frames from videos containing large faces. Optimized for laptop processing with producer-consumer architecture, GPU acceleration, and resumability. Modular design for maintainability, testing, and reusability. Runs on Linux with parallel batch processing.

## Architecture
- **Language**: Python 3.8+
- **Core Libraries**:
  - `ffmpeg-python` (video I/O, scene detection, thumbnail extraction)
  - `mediapipe` (fast face detection with GPU support)
  - `opencv-python` (image processing, quality checks, gender classification)
  - `onnxruntime-gpu` (optional: 3-10x faster inference via ONNX models)
  - `sqlite3` (built-in: progress tracking and resumability)
  - `multiprocessing` + `queue` (parallel processing with producer-consumer pattern)
- **Structure**: Modular (6 files) in a `video_extractor/` folder for clean organization.

## Modules

1. **video_processor.py**:
   - Handles video I/O: Duration/resolution retrieval, frame extraction via FFmpeg.
   - Optimizations: Embedded thumbnail extraction, scene detection, I-frame-only extraction.
   - Functions:
     - `get_video_metadata()` - duration, resolution, codec info
     - `extract_thumbnail()` - get embedded thumbnail
     - `detect_scenes()` - FFmpeg scene detection
     - `extract_iframe()` - extract single I-frame at timestamp
     - `batch_extract_frames()` - extract multiple I-frames efficiently

2. **face_analyzer.py**:
   - AI logic: Face detection (MediaPipe), quality analysis, gender classification.
   - GPU acceleration support with batching for efficiency.
   - Functions:
     - `detect_faces_batch()` - MediaPipe face detection on frame batch
     - `calculate_face_metrics()` - size, occlusion, landmark quality
     - `check_image_quality()` - blur detection via Laplacian variance
     - `detect_gender()` - gender classification model
     - `score_face()` - composite scoring (size × gender × quality)

3. **database.py**:
   - SQLite database for tracking processed videos and resumability.
   - Stores: video path, processing status, extracted frame info, timestamps.
   - Functions:
     - `init_database()` - create schema
     - `get_unprocessed_videos()` - resume support
     - `mark_video_processed()` - update status
     - `get_processing_stats()` - progress metrics

4. **utils.py**:
   - Helpers: Logging, file path handling, threshold configs, error handling.
   - Functions:
     - `setup_logging()` - structured logging with rotation
     - `validate_paths()` - input validation
     - `get_adaptive_thresholds()` - resolution-based threshold calculation
     - `retry_on_failure()` - decorator for retry logic with exponential backoff

5. **worker.py**:
   - Worker process logic for parallel processing.
   - Implements producer-consumer pattern: separate I/O from ML inference.
   - Functions:
     - `frame_extractor_worker()` - producer: extracts frames from videos
     - `face_analyzer_worker()` - consumer: processes frames with ML models

6. **main.py**:
   - Orchestrator: CLI args, adaptive algorithm loop, parallel processing coordination.
   - Entry point: Parses inputs, manages work queues, aggregates results.
   - Features:
     - `--resume` flag for checkpointing
     - `--threads N` for parallel FFmpeg
     - `--gpu` flag for GPU acceleration
     - `--dry-run` for testing

## Data Flow (Producer-Consumer Pattern)

### Sequential Flow (Single Video)
1. `main.py` → Call `video_processor.extract_thumbnail()` → Quick check
2. If no match → `video_processor.detect_scenes()` → Get keyframes
3. Adaptive loop: `video_processor.extract_iframe()` → `face_analyzer.detect_faces_batch()`
4. `face_analyzer.score_face()` → Select best → Save JPEG
5. `database.mark_video_processed()` → Update tracking

### Parallel Flow (Batch Processing)
1. **Main Process**:
   - Scan directory → Query `database.get_unprocessed_videos()`
   - Create `video_queue` (input) and `result_queue` (output)
   - Spawn N worker processes

2. **Worker Processes** (producer-consumer):
   - **I/O Workers** (2-4 processes): Extract frames → Push to `frame_queue`
   - **ML Workers** (1-2 processes): Pop from `frame_queue` → Run inference → Push results
   - Separation prevents GPU starvation and maximizes throughput

3. **Result Aggregation**:
   - Main process collects results → Saves JPEGs → Updates database

## Error Handling & Reliability

1. **Retry Logic**:
   - Exponential backoff for transient FFmpeg errors (3 retries max)
   - Corrupted video detection → Quarantine list in database

2. **Graceful Degradation**:
   - GPU unavailable → Fall back to CPU inference
   - Scene detection fails → Fall back to time-based sampling
   - No qualifying face → Save best available or skip (configurable)

3. **Monitoring**:
   - Structured logging with timestamps and video paths
   - Progress tracking via database queries
   - Optional: Export metrics to Prometheus format

## Performance Optimizations

1. **FFmpeg Optimizations**:
   - Use `--threads` flag for parallel decoding
   - I-frame-only extraction (`-skip_frame nokey`)
   - Scene detection in single pass

2. **ML Optimizations**:
   - Batch processing for GPU efficiency (3-5 frames/batch)
   - MediaPipe GPU backend when available
   - ONNX Runtime with TensorRT for 3-10x speedup

3. **Caching**:
   - SQLite database prevents reprocessing
   - Intermediate results cached (scene timestamps, extracted frames)

## Benefits
- **Speed**: 2-5x faster than naive implementation via smart sampling and GPU batching
- **Resumability**: SQLite tracking allows interrupted batch jobs to resume
- **Modularity**: Easy testing, updates, and model swapping
- **Parallelism**: Producer-consumer pattern maximizes CPU/GPU utilization
- **Robustness**: Retry logic and error handling prevent crashes

## Deployment

### Installation
```bash
# Core dependencies
pip install opencv-python ffmpeg-python mediapipe

# Optional: GPU acceleration
pip install onnxruntime-gpu

# System requirement
apt-get install ffmpeg  # Ubuntu/Debian
```

### Usage
```bash
# Basic usage
python main.py --input /videos --output /thumbs

# With resumability and GPU
python main.py --input /videos --output /thumbs --resume --gpu

# Parallel processing with 4 workers
python main.py --input /videos --output /thumbs --workers 4 --threads 2

# Dry run (test without saving)
python main.py --input /videos --dry-run
```

### Configuration
- Edit `config.yaml` for thresholds, gender preference weights, quality settings
- Environment variables: `FACETHUMB_GPU=1`, `FACETHUMB_LOG_LEVEL=DEBUG`