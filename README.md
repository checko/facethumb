# Video Face Extraction Tool

Extract representative JPEG frames from videos containing large, identifiable faces.

## Features

- **Smart Face Detection**: Uses MediaPipe for accurate face detection
- **Adaptive Sampling**: Efficiently finds best frames without processing entire videos
- **Batch Processing**: Process entire directories of videos
- **Resume Support**: Track progress and resume interrupted jobs
- **Parallel Processing**: Multi-core support for faster batch processing

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
```

### Options

- `--input, -i`: Input video file or directory (required)
- `--output, -o`: Output directory (default: same as input)
- `--threshold, -t`: Minimum face area ratio (default: 0.10)
- `--start-time, -s`: Initial timestamp to check in seconds (default: 60)
- `--log-level, -l`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--log-file`: Write logs to file

### Examples

```bash
# Process with custom threshold (5% face area)
python -m video_extractor.main -i videos/ -o output/ --threshold 0.05

# Start checking at 2 minutes into videos
python -m video_extractor.main -i videos/ --start-time 120

# Debug mode with log file
python -m video_extractor.main -i video.mp4 --log-level DEBUG --log-file debug.log
```

## Current Status

**Phase 1 (MVP)**: ✅ Complete
- Basic frame extraction
- MediaPipe face detection
- Single video and batch processing

**Upcoming Phases**:
- Phase 2: Adaptive algorithm with scene detection
- Phase 3: Gender classification and scoring
- Phase 4: Database and resume support
- Phase 5: Parallel processing
- Phase 6: Error handling and polish
- Phase 7: Configuration and optimization

See [implementation_plan.md](implementation_plan.md) for detailed roadmap.

## Project Structure

```
video_extractor/
├── __init__.py           # Package initialization
├── main.py              # CLI entry point
├── utils.py             # Logging, validation, constants
├── video_processor.py   # FFmpeg frame extraction
└── face_analyzer.py     # MediaPipe face detection
```

## Requirements

- Python 3.8+
- FFmpeg
- OpenCV
- MediaPipe

## License

MIT
