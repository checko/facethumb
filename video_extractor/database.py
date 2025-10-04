"""
Database module for tracking video processing progress.

Provides SQLite-based tracking for resumability and progress monitoring.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger("video_extractor")


class VideoDatabase:
    """SQLite database for tracking video processing status."""

    def __init__(self, db_path: str = "video_extraction.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_schema()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            logger.debug(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _create_schema(self):
        """Create database tables if they don't exist."""
        try:
            cursor = self.conn.cursor()

            # Videos table: tracks processing status
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    output_path TEXT,
                    face_area_ratio REAL,
                    score REAL,
                    timestamp_seconds REAL,
                    error_message TEXT
                )
            """)

            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON videos(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_video_path
                ON videos(video_path)
            """)

            self.conn.commit()
            logger.debug("Database schema created/verified")

        except sqlite3.Error as e:
            logger.error(f"Schema creation failed: {e}")
            raise

    def get_unprocessed_videos(self, all_videos: List[Path]) -> List[Path]:
        """
        Get list of videos that haven't been successfully processed.

        Args:
            all_videos: List of all video file paths

        Returns:
            List of video paths that need processing
        """
        try:
            cursor = self.conn.cursor()

            # Get all successfully processed videos
            cursor.execute("""
                SELECT video_path
                FROM videos
                WHERE status = 'success'
            """)

            processed_paths = {row['video_path'] for row in cursor.fetchall()}

            # Filter out already processed videos
            unprocessed = [
                video for video in all_videos
                if str(video) not in processed_paths
            ]

            logger.debug(
                f"Found {len(unprocessed)} unprocessed videos "
                f"out of {len(all_videos)} total"
            )

            return unprocessed

        except sqlite3.Error as e:
            logger.error(f"Failed to get unprocessed videos: {e}")
            # On error, return all videos (fail-safe)
            return all_videos

    def mark_video_processing(self, video_path: Path):
        """
        Mark video as currently being processed.

        Args:
            video_path: Path to video file
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO videos
                (video_path, status, updated_at)
                VALUES (?, 'processing', CURRENT_TIMESTAMP)
            """, (str(video_path),))

            self.conn.commit()
            logger.debug(f"Marked as processing: {video_path.name}")

        except sqlite3.Error as e:
            logger.error(f"Failed to mark video as processing: {e}")

    def mark_video_processed(
        self,
        video_path: Path,
        success: bool,
        output_path: Optional[Path] = None,
        face_area_ratio: Optional[float] = None,
        score: Optional[float] = None,
        timestamp_seconds: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        """
        Mark video as processed (success or failure).

        Args:
            video_path: Path to video file
            success: Whether processing was successful
            output_path: Path to output JPEG (if successful)
            face_area_ratio: Face area ratio of extracted frame
            score: Composite score of extracted frame
            timestamp_seconds: Timestamp of extracted frame
            error_message: Error message (if failed)
        """
        try:
            cursor = self.conn.cursor()

            status = 'success' if success else 'failed'

            cursor.execute("""
                INSERT OR REPLACE INTO videos
                (video_path, status, processed_at, output_path,
                 face_area_ratio, score, timestamp_seconds, error_message, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                str(video_path),
                status,
                str(output_path) if output_path else None,
                face_area_ratio,
                score,
                timestamp_seconds,
                error_message
            ))

            self.conn.commit()
            logger.debug(f"Marked as {status}: {video_path.name}")

        except sqlite3.Error as e:
            logger.error(f"Failed to mark video as processed: {e}")

    def get_processing_stats(self) -> Dict[str, any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with stats: total, success, failed, processing, pending
        """
        try:
            cursor = self.conn.cursor()

            # Get counts by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM videos
                GROUP BY status
            """)

            stats = {
                'total': 0,
                'success': 0,
                'failed': 0,
                'processing': 0
            }

            for row in cursor.fetchall():
                status = row['status']
                count = row['count']
                stats[status] = count
                stats['total'] += count

            return stats

        except sqlite3.Error as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {'total': 0, 'success': 0, 'failed': 0, 'processing': 0}

    def get_video_status(self, video_path: Path) -> Optional[str]:
        """
        Get processing status of a specific video.

        Args:
            video_path: Path to video file

        Returns:
            Status string ('success', 'failed', 'processing') or None if not found
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                SELECT status
                FROM videos
                WHERE video_path = ?
            """, (str(video_path),))

            row = cursor.fetchone()
            return row['status'] if row else None

        except sqlite3.Error as e:
            logger.error(f"Failed to get video status: {e}")
            return None

    def reset_processing_videos(self):
        """
        Reset videos stuck in 'processing' state.

        Useful for cleaning up after interrupted runs.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                UPDATE videos
                SET status = 'failed',
                    error_message = 'Processing interrupted',
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = 'processing'
            """)

            count = cursor.rowcount
            self.conn.commit()

            if count > 0:
                logger.info(f"Reset {count} interrupted video(s)")

        except sqlite3.Error as e:
            logger.error(f"Failed to reset processing videos: {e}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
