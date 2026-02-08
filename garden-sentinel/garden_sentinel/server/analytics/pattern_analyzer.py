"""
Predator pattern learning and prediction.

Analyzes historical detection data to:
- Identify when specific predators typically visit
- Predict likely attack windows
- Alert proactively before expected visits
- Track effectiveness of deterrents
"""

import asyncio
import logging
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


class DayOfWeek(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class TimeWindow:
    """A time window during the day."""
    start_hour: int
    start_minute: int = 0
    end_hour: int = 0
    end_minute: int = 0

    def __post_init__(self):
        if self.end_hour == 0 and self.end_minute == 0:
            # Default to 1 hour window
            self.end_hour = self.start_hour + 1
            if self.end_hour >= 24:
                self.end_hour = 23
                self.end_minute = 59

    @property
    def start_time(self) -> dt_time:
        return dt_time(self.start_hour, self.start_minute)

    @property
    def end_time(self) -> dt_time:
        return dt_time(self.end_hour, self.end_minute)

    def contains(self, t: dt_time) -> bool:
        """Check if a time falls within this window."""
        start = self.start_hour * 60 + self.start_minute
        end = self.end_hour * 60 + self.end_minute
        check = t.hour * 60 + t.minute
        return start <= check <= end

    def __str__(self) -> str:
        return f"{self.start_hour:02d}:{self.start_minute:02d}-{self.end_hour:02d}:{self.end_minute:02d}"


@dataclass
class PredatorPattern:
    """Learned pattern for a specific predator type."""
    predator_type: str
    total_visits: int = 0

    # Time-of-day distribution (24 hourly buckets)
    hourly_counts: list[int] = field(default_factory=lambda: [0] * 24)

    # Day-of-week distribution
    daily_counts: list[int] = field(default_factory=lambda: [0] * 7)

    # Peak visit windows
    peak_windows: list[TimeWindow] = field(default_factory=list)

    # Average visit duration (seconds)
    avg_visit_duration: float = 0.0

    # Deterrent effectiveness (sprayed visits that left quickly)
    spray_effectiveness: float = 0.0
    visits_sprayed: int = 0
    visits_deterred: int = 0  # Left within 30s of spray

    # Location hotspots (grid cells with high activity)
    location_hotspots: list[tuple[float, float]] = field(default_factory=list)

    # Recent trend (visits per day, last 7 days vs previous 7 days)
    recent_trend: float = 0.0  # positive = increasing

    # Last seen
    last_visit: Optional[datetime] = None

    def get_risk_score(self, current_time: datetime) -> float:
        """
        Calculate current risk score (0-1) based on patterns.

        Higher score = higher likelihood of visit.
        """
        if self.total_visits == 0:
            return 0.0

        score = 0.0

        # Time-of-day factor
        hour = current_time.hour
        max_hourly = max(self.hourly_counts) if self.hourly_counts else 1
        if max_hourly > 0:
            hour_factor = self.hourly_counts[hour] / max_hourly
            score += hour_factor * 0.4

        # Day-of-week factor
        day = current_time.weekday()
        max_daily = max(self.daily_counts) if self.daily_counts else 1
        if max_daily > 0:
            day_factor = self.daily_counts[day] / max_daily
            score += day_factor * 0.2

        # Peak window factor
        current_time_only = current_time.time()
        in_peak = any(w.contains(current_time_only) for w in self.peak_windows)
        if in_peak:
            score += 0.3

        # Recent trend factor
        if self.recent_trend > 0:
            score += min(0.1, self.recent_trend * 0.05)

        return min(1.0, score)


@dataclass
class VisitPrediction:
    """Prediction for upcoming predator visit."""
    predator_type: str
    predicted_window: TimeWindow
    confidence: float  # 0-1
    risk_score: float  # 0-1
    historical_visits_in_window: int
    message: str


class PatternAnalyzer:
    """
    Analyzes predator detection patterns and makes predictions.

    Uses SQLite database for persistent storage of detection events.
    """

    def __init__(
        self,
        db_path: Path,
        prediction_lookahead_hours: int = 2,
        min_visits_for_pattern: int = 3,
    ):
        self.db_path = db_path
        self.prediction_lookahead_hours = prediction_lookahead_hours
        self.min_visits_for_pattern = min_visits_for_pattern

        # Cached patterns
        self._patterns: dict[str, PredatorPattern] = {}

        # Callbacks
        self._on_prediction: Optional[Callable[[VisitPrediction], Awaitable[None]]] = None

        # Background task
        self._prediction_task: Optional[asyncio.Task] = None

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    predator_type TEXT NOT NULL,
                    device_id TEXT,
                    confidence REAL,
                    bbox_x REAL,
                    bbox_y REAL,
                    world_x REAL,
                    world_y REAL,
                    was_sprayed BOOLEAN DEFAULT FALSE,
                    duration_seconds REAL,
                    deterred BOOLEAN DEFAULT FALSE
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_timestamp
                ON detections(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_predator
                ON detections(predator_type)
            """)

            # Table for tracking visits (grouped detections)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    predator_type TEXT NOT NULL,
                    detection_count INTEGER DEFAULT 1,
                    was_sprayed BOOLEAN DEFAULT FALSE,
                    was_deterred BOOLEAN DEFAULT FALSE,
                    primary_device_id TEXT,
                    world_x REAL,
                    world_y REAL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_visits_time
                ON visits(start_time)
            """)

    async def start(self):
        """Start background prediction task."""
        self._prediction_task = asyncio.create_task(self._prediction_loop())
        # Load existing patterns
        await self._refresh_patterns()
        logger.info("Pattern analyzer started")

    async def stop(self):
        """Stop the analyzer."""
        if self._prediction_task:
            self._prediction_task.cancel()
            try:
                await self._prediction_task
            except asyncio.CancelledError:
                pass
        logger.info("Pattern analyzer stopped")

    def set_prediction_callback(
        self,
        callback: Callable[[VisitPrediction], Awaitable[None]],
    ):
        """Set callback for predictions."""
        self._on_prediction = callback

    async def record_detection(
        self,
        predator_type: str,
        timestamp: Optional[datetime] = None,
        device_id: Optional[str] = None,
        confidence: float = 0.0,
        bbox_x: Optional[float] = None,
        bbox_y: Optional[float] = None,
        world_x: Optional[float] = None,
        world_y: Optional[float] = None,
    ):
        """Record a new detection."""
        if timestamp is None:
            timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO detections (
                    timestamp, predator_type, device_id, confidence,
                    bbox_x, bbox_y, world_x, world_y
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat(),
                predator_type,
                device_id,
                confidence,
                bbox_x,
                bbox_y,
                world_x,
                world_y,
            ))

            # Check if this is part of an existing visit or new visit
            # (within 5 minutes of last detection of same type)
            cursor = conn.execute("""
                SELECT id, end_time FROM visits
                WHERE predator_type = ?
                AND datetime(end_time) > datetime(?, '-5 minutes')
                ORDER BY end_time DESC LIMIT 1
            """, (predator_type, timestamp.isoformat()))

            row = cursor.fetchone()
            if row:
                # Update existing visit
                visit_id = row[0]
                conn.execute("""
                    UPDATE visits
                    SET end_time = ?, detection_count = detection_count + 1
                    WHERE id = ?
                """, (timestamp.isoformat(), visit_id))
            else:
                # New visit
                conn.execute("""
                    INSERT INTO visits (
                        start_time, end_time, predator_type,
                        primary_device_id, world_x, world_y
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                    predator_type,
                    device_id,
                    world_x,
                    world_y,
                ))

    async def record_spray_event(
        self,
        predator_type: str,
        timestamp: Optional[datetime] = None,
    ):
        """Record that a predator was sprayed."""
        if timestamp is None:
            timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            # Mark recent visit as sprayed
            conn.execute("""
                UPDATE visits
                SET was_sprayed = TRUE
                WHERE predator_type = ?
                AND datetime(end_time) > datetime(?, '-1 minute')
            """, (predator_type, timestamp.isoformat()))

    async def record_deterrence(
        self,
        predator_type: str,
        timestamp: Optional[datetime] = None,
    ):
        """Record that a predator was successfully deterred (left quickly after spray)."""
        if timestamp is None:
            timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE visits
                SET was_deterred = TRUE
                WHERE predator_type = ?
                AND was_sprayed = TRUE
                AND datetime(end_time) > datetime(?, '-2 minutes')
            """, (predator_type, timestamp.isoformat()))

    async def _refresh_patterns(self):
        """Refresh cached patterns from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get all predator types
            cursor = conn.execute("""
                SELECT DISTINCT predator_type FROM visits
            """)
            predator_types = [row[0] for row in cursor.fetchall()]

            for predator_type in predator_types:
                pattern = await self._analyze_predator_pattern(conn, predator_type)
                self._patterns[predator_type] = pattern

        logger.info(f"Refreshed patterns for {len(self._patterns)} predator types")

    async def _analyze_predator_pattern(
        self,
        conn: sqlite3.Connection,
        predator_type: str,
    ) -> PredatorPattern:
        """Analyze patterns for a specific predator type."""
        pattern = PredatorPattern(predator_type=predator_type)

        # Total visits
        cursor = conn.execute("""
            SELECT COUNT(*) FROM visits WHERE predator_type = ?
        """, (predator_type,))
        pattern.total_visits = cursor.fetchone()[0]

        # Hourly distribution
        cursor = conn.execute("""
            SELECT CAST(strftime('%H', start_time) AS INTEGER) as hour, COUNT(*)
            FROM visits WHERE predator_type = ?
            GROUP BY hour
        """, (predator_type,))
        for row in cursor:
            pattern.hourly_counts[row[0]] = row[1]

        # Daily distribution
        cursor = conn.execute("""
            SELECT CAST(strftime('%w', start_time) AS INTEGER) as day, COUNT(*)
            FROM visits WHERE predator_type = ?
            GROUP BY day
        """, (predator_type,))
        for row in cursor:
            # SQLite %w: 0=Sunday, we want 0=Monday
            day_idx = (row[0] - 1) % 7
            pattern.daily_counts[day_idx] = row[1]

        # Find peak windows (hours with above-average visits)
        if pattern.total_visits > 0:
            avg_hourly = pattern.total_visits / 24
            peak_hours = [
                h for h, count in enumerate(pattern.hourly_counts)
                if count > avg_hourly * 1.5
            ]

            # Group consecutive hours into windows
            if peak_hours:
                windows = []
                start = peak_hours[0]
                end = start

                for h in peak_hours[1:]:
                    if h == end + 1:
                        end = h
                    else:
                        windows.append(TimeWindow(start, 0, end, 59))
                        start = h
                        end = h

                windows.append(TimeWindow(start, 0, end, 59))
                pattern.peak_windows = windows[:3]  # Top 3 windows

        # Average visit duration
        cursor = conn.execute("""
            SELECT AVG(
                (julianday(end_time) - julianday(start_time)) * 86400
            ) FROM visits
            WHERE predator_type = ? AND end_time IS NOT NULL
        """, (predator_type,))
        result = cursor.fetchone()[0]
        pattern.avg_visit_duration = result if result else 0.0

        # Spray effectiveness
        cursor = conn.execute("""
            SELECT
                COUNT(*) as sprayed,
                SUM(CASE WHEN was_deterred THEN 1 ELSE 0 END) as deterred
            FROM visits
            WHERE predator_type = ? AND was_sprayed = TRUE
        """, (predator_type,))
        row = cursor.fetchone()
        pattern.visits_sprayed = row[0] or 0
        pattern.visits_deterred = row[1] or 0
        if pattern.visits_sprayed > 0:
            pattern.spray_effectiveness = pattern.visits_deterred / pattern.visits_sprayed

        # Recent trend (last 7 days vs previous 7 days)
        cursor = conn.execute("""
            SELECT
                SUM(CASE WHEN date(start_time) >= date('now', '-7 days') THEN 1 ELSE 0 END) as recent,
                SUM(CASE WHEN date(start_time) >= date('now', '-14 days')
                         AND date(start_time) < date('now', '-7 days') THEN 1 ELSE 0 END) as previous
            FROM visits WHERE predator_type = ?
        """, (predator_type,))
        row = cursor.fetchone()
        recent = row[0] or 0
        previous = row[1] or 0
        if previous > 0:
            pattern.recent_trend = (recent - previous) / previous
        elif recent > 0:
            pattern.recent_trend = 1.0  # New activity

        # Last visit
        cursor = conn.execute("""
            SELECT MAX(start_time) FROM visits WHERE predator_type = ?
        """, (predator_type,))
        result = cursor.fetchone()[0]
        if result:
            pattern.last_visit = datetime.fromisoformat(result)

        # Location hotspots (if we have world coordinates)
        cursor = conn.execute("""
            SELECT world_x, world_y, COUNT(*) as count
            FROM visits
            WHERE predator_type = ?
            AND world_x IS NOT NULL AND world_y IS NOT NULL
            GROUP BY ROUND(world_x, 0), ROUND(world_y, 0)
            ORDER BY count DESC
            LIMIT 5
        """, (predator_type,))
        pattern.location_hotspots = [
            (row[0], row[1]) for row in cursor
        ]

        return pattern

    async def _prediction_loop(self):
        """Background loop to generate predictions."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._check_predictions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")

    async def _check_predictions(self):
        """Check if any predictions should be generated."""
        now = datetime.now()

        for predator_type, pattern in self._patterns.items():
            if pattern.total_visits < self.min_visits_for_pattern:
                continue

            # Check if we're approaching a peak window
            for window in pattern.peak_windows:
                window_start = datetime.combine(now.date(), window.start_time)

                # If window starts within lookahead period and we haven't warned yet
                time_until_window = (window_start - now).total_seconds() / 3600

                if 0 < time_until_window <= self.prediction_lookahead_hours:
                    risk_score = pattern.get_risk_score(window_start)

                    if risk_score > 0.5:
                        prediction = VisitPrediction(
                            predator_type=predator_type,
                            predicted_window=window,
                            confidence=min(1.0, pattern.total_visits / 20),
                            risk_score=risk_score,
                            historical_visits_in_window=sum(
                                pattern.hourly_counts[window.start_hour:window.end_hour + 1]
                            ),
                            message=self._build_prediction_message(
                                predator_type, pattern, window, time_until_window
                            ),
                        )

                        logger.info(
                            f"Prediction: {predator_type} likely in {time_until_window:.1f}h "
                            f"(risk: {risk_score:.0%})"
                        )

                        if self._on_prediction:
                            await self._on_prediction(prediction)

    def _build_prediction_message(
        self,
        predator_type: str,
        pattern: PredatorPattern,
        window: TimeWindow,
        hours_until: float,
    ) -> str:
        """Build a human-readable prediction message."""
        parts = [
            f"⚠️ {predator_type.title()} activity expected",
            f"around {window.start_time.strftime('%H:%M')}",
            f"({hours_until:.0f}h from now).",
        ]

        if pattern.total_visits >= 10:
            parts.append(f"Based on {pattern.total_visits} historical visits.")

        if pattern.spray_effectiveness > 0.5:
            parts.append(
                f"Water deterrent has been {pattern.spray_effectiveness:.0%} effective."
            )

        return " ".join(parts)

    def get_pattern(self, predator_type: str) -> Optional[PredatorPattern]:
        """Get pattern for a specific predator type."""
        return self._patterns.get(predator_type)

    def get_all_patterns(self) -> dict[str, PredatorPattern]:
        """Get all learned patterns."""
        return self._patterns.copy()

    def get_current_risk_levels(self) -> dict[str, float]:
        """Get current risk levels for all predator types."""
        now = datetime.now()
        return {
            predator_type: pattern.get_risk_score(now)
            for predator_type, pattern in self._patterns.items()
        }

    async def get_statistics(self) -> dict:
        """Get overall statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Total visits
            cursor = conn.execute("SELECT COUNT(*) FROM visits")
            stats["total_visits"] = cursor.fetchone()[0]

            # Visits by predator type
            cursor = conn.execute("""
                SELECT predator_type, COUNT(*) FROM visits
                GROUP BY predator_type ORDER BY COUNT(*) DESC
            """)
            stats["visits_by_type"] = dict(cursor.fetchall())

            # Visits today
            cursor = conn.execute("""
                SELECT COUNT(*) FROM visits
                WHERE date(start_time) = date('now')
            """)
            stats["visits_today"] = cursor.fetchone()[0]

            # Visits this week
            cursor = conn.execute("""
                SELECT COUNT(*) FROM visits
                WHERE date(start_time) >= date('now', '-7 days')
            """)
            stats["visits_this_week"] = cursor.fetchone()[0]

            # Overall deterrence rate
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as sprayed,
                    SUM(CASE WHEN was_deterred THEN 1 ELSE 0 END) as deterred
                FROM visits WHERE was_sprayed = TRUE
            """)
            row = cursor.fetchone()
            if row[0] and row[0] > 0:
                stats["deterrence_rate"] = row[1] / row[0]
            else:
                stats["deterrence_rate"] = 0.0

            # Peak hours
            cursor = conn.execute("""
                SELECT CAST(strftime('%H', start_time) AS INTEGER) as hour, COUNT(*) as count
                FROM visits GROUP BY hour ORDER BY count DESC LIMIT 3
            """)
            stats["peak_hours"] = [
                {"hour": row[0], "visits": row[1]}
                for row in cursor
            ]

            return stats

    async def get_daily_breakdown(
        self,
        days: int = 7,
    ) -> list[dict]:
        """Get visit breakdown by day."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    date(start_time) as day,
                    predator_type,
                    COUNT(*) as visits
                FROM visits
                WHERE date(start_time) >= date('now', ?)
                GROUP BY day, predator_type
                ORDER BY day DESC
            """, (f'-{days} days',))

            # Organize by day
            daily = defaultdict(lambda: {"date": None, "total": 0, "by_type": {}})
            for row in cursor:
                day_str = row[0]
                daily[day_str]["date"] = day_str
                daily[day_str]["total"] += row[2]
                daily[day_str]["by_type"][row[1]] = row[2]

            return list(daily.values())
