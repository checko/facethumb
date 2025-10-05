# Video Face Extraction Tool (FaceThumb)

Extract representative JPEG frames from videos containing large, identifiable faces. Optimized for 100% success rate with intelligent adaptive sampling.

## Features

- **100% Output Guarantee**: Always extracts a frame from every video with intelligent fallback strategies
- **Smart Face Detection**: MediaPipe with GPU acceleration for fast, accurate detection
- **Adaptive Sampling**: Intelligently samples 16 timestamps (every 5%) across video duration
- **Fallback Strategy**: No face found → use biggest face or first frame (ensures output for all videos)
- **Gender Preference**: Optional gender classification with configurable weighting
- **Resume Support**: SQLite-based progress tracking for interrupted batch jobs
- **Quality Scoring**: Composite scoring (size 45%, quality 27%, confidence 18%, gender 10%)
- **Batch Processing**: Process entire directories with automatic database tracking

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg installed on system

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Process single video
python -m video_extractor.main --input video.mp4 --output ./output

# Process directory of videos
python -m video_extractor.main --input /path/to/videos --output ./output

# Resume interrupted processing
python -m video_extractor.main --input /path/to/videos --output ./output --resume
```

### Options

**Input/Output:**
- `--input, -i`: Input video file or directory (required)
- `--output, -o`: Output directory (default: same folder as each video file)

**Face Detection:**
- `--threshold, -t`: Minimum face area ratio (default: 0.02 = 2%)
- `--start-time, -s`: Initial timestamp to check in seconds (default: 60)

**Gender Preference:**
- `--gender-preference, -g`: Prefer faces of specific gender: female, male, none (default: female)
- `--gender-weight`: Weight for gender preference 0.0-1.0 (default: 0.1)

**Database & Resume:**
- `--resume`: Resume previous processing session (skip already processed videos)
- `--db-path`: Path to database file (default: video_extraction.db)

**Logging:**
- `--log-level, -l`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--log-file`: Write logs to file

### Examples

```bash
# Process with lower threshold for smaller faces
python -m video_extractor.main -i videos/ -o output/ --threshold 0.01

# Resume interrupted batch job
python -m video_extractor.main -i videos/ -o output/ --resume

# Prefer male faces with higher weight
python -m video_extractor.main -i videos/ -g male --gender-weight 0.2

# Debug mode with log file
python -m video_extractor.main -i video.mp4 --log-level DEBUG --log-file debug.log

# No gender preference
python -m video_extractor.main -i videos/ -g none
```

## Current Status

**Phase 1 (MVP)**: Complete
- Basic frame extraction with MediaPipe
- Single video and batch processing
- Bug fixes: subprocess conflict, false positive filtering

**Phase 2 (Adaptive Algorithm)**: Complete
- Dense adaptive sampling (16 timestamps every 5%)
- Thumbnail extraction (fast check)
- Quality scoring with blur detection
- 100% success rate achieved on test set

**Phase 3 (Gender Classification)**: Complete
- Gender detection framework (placeholder implementation)
- Gender-aware composite scoring
- CLI flags for gender preference and weight
- Configurable scoring weights

**Phase 4 (Database & Resumability)**: Complete
- SQLite-based progress tracking
- Resume interrupted processing sessions
- Skip already processed videos
- Detailed metadata storage (status, face_area, score, timestamp)

**Phase 5 (Parallel Processing)**: Pending
- Multi-process worker pools
- Producer-consumer pattern

**Phase 6 (Error Handling)**: Pending
- Retry logic with exponential backoff
- GPU fallback to CPU

**Phase 7 (Optimization)**: Pending
- YAML configuration file
- ONNX Runtime integration

See [implementation_plan.md](implementation_plan.md) and [progress.md](progress.md) for detailed roadmap.

## Project Structure

```
video_extractor/
├── __init__.py           # Package initialization
├── main.py              # CLI entry point and adaptive algorithm
├── utils.py             # Logging, validation, constants
├── video_processor.py   # FFmpeg frame extraction, thumbnail extraction
├── face_analyzer.py     # MediaPipe face detection, quality scoring, gender detection
└── database.py          # SQLite progress tracking and resumability
```

## Performance

- **Output Rate**: 100% (every video gets a JPG output)
- **Speed**: ~10 seconds per video (with dense sampling)
- **Sampling**: 16 timestamps across video duration
- **Fallback**: Automatic fallback to biggest face or first frame when no qualifying faces found

## Configuration

Default values optimized for best quality with guaranteed output:

```python
DEFAULT_LARGE_FACE_THRESHOLD = 0.02  # 2% of frame area
DEFAULT_SAMPLING_PERCENTAGES = [0.05, 0.10, 0.15, ... 0.80]  # 16 samples
DEFAULT_MAX_SAMPLES = 16
DEFAULT_GENDER_PREFERENCE = 'female'
DEFAULT_GENDER_WEIGHT = 0.1
```

**Fallback Strategy:**
1. No qualifying faces (>2% threshold) → Use biggest face detected
2. No faces detected at all → Use first extracted frame
3. Video too short (<5s) → Use first frame

## Requirements

- Python 3.8+
- FFmpeg
- opencv-python
- mediapipe
- ffmpeg-python
- numpy

## License

MIT
