# Implementation Plan: Video Face Extraction Tool

## Strategy: Iterative Vertical Slices

Building the system in incremental phases, each delivering a working version with progressively more features. Each phase is tested and committed before moving to the next.

---

## Phase 1: Minimal Viable Version (MVP)
**Goal**: Extract one frame from one video with basic face detection
**Time**: 3-4 hours
**Status**: 🔄 In Progress

### Modules to implement:
1. **utils.py** (30 min)
   - Basic logging setup
   - Path validation
   - Simple config constants

2. **video_processor.py** - Basic version (45 min)
   - `get_video_metadata()` - duration and resolution only
   - `extract_iframe()` - extract single frame at timestamp

3. **face_analyzer.py** - Basic version (1 hour)
   - `detect_faces_batch()` - MediaPipe face detection
   - `calculate_face_metrics()` - just bounding box size

4. **main.py** - Simple CLI (45 min)
   - Argparse for `--input` and `--output`
   - Process single video with fixed timestamp (1 minute)
   - Save if face found

### Test:
- Run on 5-10 test videos
- Verify it extracts faces

### Git commit:
"MVP: Basic single-video face extraction"

---

## Phase 2: Add Adaptive Algorithm
**Goal**: Smart sampling to find best frame
**Time**: 2-3 hours
**Status**: ⏸️ Pending

### Enhancements:
1. **video_processor.py** additions:
   - `extract_thumbnail()` - check embedded thumbnail first
   - `detect_scenes()` - FFmpeg scene detection

2. **face_analyzer.py** additions:
   - `check_image_quality()` - blur detection
   - `score_face()` - composite scoring

3. **main.py** - Implement adaptive loop:
   - Stage 1: Thumbnail check
   - Stage 2: Scene detection
   - Stage 3: Adaptive time-based sampling

### Test:
- Compare results with Phase 1
- Measure processing time improvement

### Git commit:
"Add adaptive sampling algorithm"

---

## Phase 3: Gender Classification & Scoring
**Goal**: Prefer female faces, score-based selection
**Time**: 2 hours
**Status**: ⏸️ Pending

### Enhancements:
1. **face_analyzer.py**:
   - `detect_gender()` - integrate pre-trained gender model
   - Update `score_face()` with gender weighting

2. **utils.py**:
   - `get_adaptive_thresholds()` - resolution-based thresholds

### Test:
- Verify gender detection works
- Test scoring logic

### Git commit:
"Add gender classification and enhanced scoring"

---

## Phase 4: Batch Processing & Database
**Goal**: Process directories with resumability
**Time**: 2-3 hours
**Status**: ⏸️ Pending

### New modules:
1. **database.py** (1 hour)
   - SQLite schema
   - All database functions

2. **main.py** enhancements:
   - Directory scanning
   - Database integration
   - `--resume` flag

### Test:
- Process 50-100 videos
- Test resume functionality

### Git commit:
"Add batch processing and resumability"

---

## Phase 5: Parallel Processing
**Goal**: Speed up with multiprocessing
**Time**: 2-3 hours
**Status**: ⏸️ Pending

### New module:
1. **worker.py** (1.5 hours)
   - Producer-consumer pattern
   - Frame extractor workers
   - Face analyzer workers

2. **main.py** enhancements:
   - Multi-process coordination
   - Queue management
   - `--workers` flag

### Test:
- Benchmark single-process vs multi-process
- Verify no race conditions

### Git commit:
"Add parallel processing with worker pools"

---

## Phase 6: Error Handling & Polish
**Goal**: Production-ready reliability
**Time**: 3-4 hours
**Status**: ⏸️ Pending

### Enhancements across modules:
1. **utils.py**:
   - `retry_on_failure()` decorator
   - Enhanced logging with rotation

2. **All modules**:
   - Try-catch blocks
   - Graceful degradation
   - Input validation

3. **main.py**:
   - `--gpu`, `--threads`, `--dry-run` flags
   - Progress reporting

### Test:
- Throw corrupted videos, missing files, etc. at it

### Git commit:
"Add error handling and production polish"

---

## Phase 7: Optimization & Configuration
**Goal**: Fine-tune performance
**Time**: 2-3 hours
**Status**: ⏸️ Pending

### Additions:
1. **config.yaml** - Externalized configuration
2. **Optional ONNX Runtime** integration
3. Performance profiling and optimization

### Test:
- Benchmark on large dataset (500+ videos)
- Optimize bottlenecks

### Git commit:
"Add configuration system and performance optimizations"

---

## Total Timeline
**Estimated**: 20-25 hours of focused work

## Benefits of This Approach
- ✅ Working software after Phase 1 (3-4 hours)
- ✅ Real feedback early in development
- ✅ Each phase is testable end-to-end
- ✅ Can adjust based on real performance data
- ✅ Lower risk than building everything then integrating
