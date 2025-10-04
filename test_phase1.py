#!/usr/bin/env python3
"""
Phase 1 Testing Script

Tests the MVP implementation of the Video Face Extraction Tool.
"""

import subprocess
import sys
from pathlib import Path
import shutil
import json
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_info(text):
    """Print info message."""
    print(f"  {text}")


def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Checking Dependencies")

    all_ok = True

    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print_success(f"FFmpeg installed: {version}")
        else:
            print_error("FFmpeg not working properly")
            all_ok = False
    except FileNotFoundError:
        print_error("FFmpeg not found. Install with: sudo apt-get install ffmpeg")
        all_ok = False
    except Exception as e:
        print_error(f"Error checking FFmpeg: {e}")
        all_ok = False

    # Check Python packages
    packages = ['cv2', 'mediapipe', 'numpy', 'ffmpeg']
    for package in packages:
        try:
            __import__(package)
            print_success(f"Python package '{package}' installed")
        except ImportError:
            print_error(f"Python package '{package}' not found. Run: pip install -r requirements.txt")
            all_ok = False

    return all_ok


def check_test_videos():
    """Check if test videos exist."""
    print_header("Checking Test Videos")

    test_dir = Path('testvideo')

    if not test_dir.exists():
        print_warning(f"Test directory '{test_dir}' does not exist")
        print_info("Creating testvideo/ directory...")
        test_dir.mkdir(exist_ok=True)
        print_info("Please add some test videos (.mp4, .avi, .mov) to testvideo/")
        return []

    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    videos = []
    for ext in video_extensions:
        videos.extend(list(test_dir.glob(f'*{ext}')))
        videos.extend(list(test_dir.glob(f'*{ext.upper()}')))

    if videos:
        print_success(f"Found {len(videos)} test video(s):")
        for video in videos:
            size_mb = video.stat().st_size / (1024 * 1024)
            print_info(f"  - {video.name} ({size_mb:.1f} MB)")
    else:
        print_warning("No test videos found in testvideo/")
        print_info("Please add some test videos (.mp4, .avi, .mov) to testvideo/")

    return videos


def run_test(name, command, expected_success=True):
    """Run a single test command."""
    print(f"\n{Colors.BOLD}Test: {name}{Colors.RESET}")
    print_info(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Print output
        if result.stdout:
            print_info("Output:")
            for line in result.stdout.strip().split('\n'):
                print_info(f"  {line}")

        if result.stderr and result.returncode != 0:
            print_info("Errors:")
            for line in result.stderr.strip().split('\n'):
                print_info(f"  {line}")

        # Check result
        if expected_success:
            if result.returncode == 0:
                print_success(f"Test passed: {name}")
                return True
            else:
                print_error(f"Test failed: {name} (exit code {result.returncode})")
                return False
        else:
            if result.returncode != 0:
                print_success(f"Test passed: {name} (expected failure)")
                return True
            else:
                print_error(f"Test failed: {name} (expected to fail but succeeded)")
                return False

    except subprocess.TimeoutExpired:
        print_error(f"Test timeout: {name}")
        return False
    except Exception as e:
        print_error(f"Test error: {name} - {e}")
        return False


def verify_output(output_dir, expected_count=None):
    """Verify output files were created."""
    print(f"\n{Colors.BOLD}Verifying Output{Colors.RESET}")

    output_path = Path(output_dir)
    if not output_path.exists():
        print_error(f"Output directory does not exist: {output_dir}")
        return False

    jpgs = list(output_path.glob('*.jpg'))

    if jpgs:
        print_success(f"Found {len(jpgs)} output JPEG(s):")
        for jpg in jpgs:
            size_kb = jpg.stat().st_size / 1024
            print_info(f"  - {jpg.name} ({size_kb:.1f} KB)")

        if expected_count is not None:
            if len(jpgs) == expected_count:
                print_success(f"Output count matches expected: {expected_count}")
            else:
                print_warning(f"Expected {expected_count} files, got {len(jpgs)}")

        return True
    else:
        print_warning("No JPEG files found in output directory")
        return False


def run_all_tests(videos):
    """Run all Phase 1 tests."""
    print_header("Running Phase 1 Tests")

    if not videos:
        print_error("Cannot run tests without video files")
        return

    # Prepare output directory
    output_dir = Path('test_output')
    if output_dir.exists():
        print_info(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    results = []

    # Test 1: Single video processing
    if len(videos) >= 1:
        test_video = videos[0]
        result = run_test(
            "Single Video Processing",
            ['python', '-m', 'video_extractor.main',
             '--input', str(test_video),
             '--output', str(output_dir),
             '--log-level', 'DEBUG']
        )
        results.append(("Single Video", result))
        verify_output(output_dir, expected_count=1)

    # Test 2: Directory processing
    if len(videos) > 1:
        # Clean output
        shutil.rmtree(output_dir)
        output_dir.mkdir()

        result = run_test(
            "Directory Processing",
            ['python', '-m', 'video_extractor.main',
             '--input', 'testvideo',
             '--output', str(output_dir),
             '--log-level', 'INFO']
        )
        results.append(("Directory Processing", result))
        verify_output(output_dir, expected_count=len(videos))

    # Test 3: Custom threshold (more sensitive)
    shutil.rmtree(output_dir)
    output_dir.mkdir()

    result = run_test(
        "Custom Threshold (0.05)",
        ['python', '-m', 'video_extractor.main',
         '--input', str(videos[0]),
         '--output', str(output_dir),
         '--threshold', '0.05',
         '--log-level', 'INFO']
    )
    results.append(("Custom Threshold", result))
    verify_output(output_dir)

    # Test 4: Custom start time
    shutil.rmtree(output_dir)
    output_dir.mkdir()

    result = run_test(
        "Custom Start Time (30s)",
        ['python', '-m', 'video_extractor.main',
         '--input', str(videos[0]),
         '--output', str(output_dir),
         '--start-time', '30',
         '--log-level', 'INFO']
    )
    results.append(("Custom Start Time", result))
    verify_output(output_dir)

    # Test 5: Invalid input (should fail gracefully)
    result = run_test(
        "Invalid Input Handling",
        ['python', '-m', 'video_extractor.main',
         '--input', 'nonexistent_video.mp4',
         '--output', str(output_dir)],
        expected_success=False
    )
    results.append(("Invalid Input", result))

    # Print summary
    print_header("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}")

    if passed == total:
        print_success("\n🎉 All Phase 1 tests passed!")
        print_info("You can proceed to Phase 2: Adaptive Algorithm")
        return True
    else:
        print_warning(f"\n⚠ {total - passed} test(s) failed")
        print_info("Please fix issues before proceeding to Phase 2")
        return False


def main():
    """Main test runner."""
    print_header("Phase 1 MVP Testing Suite")
    print_info(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check dependencies
    if not check_dependencies():
        print_error("\nDependency check failed. Please install missing dependencies.")
        print_info("Run: pip install -r requirements.txt")
        print_info("Run: sudo apt-get install ffmpeg")
        sys.exit(1)

    # Check test videos
    videos = check_test_videos()

    if not videos:
        print_warning("\nNo test videos found. Please add videos to testvideo/ directory.")
        print_info("Then run this script again: python test_phase1.py")
        sys.exit(0)

    # Run tests
    success = run_all_tests(videos)

    print_info(f"\nTest finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
