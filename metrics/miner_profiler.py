"""
Miner Profiler Module
Creates comprehensive profiles of miner behavior and historical performance
"""

import time
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Miner risk classification"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrendDirection(Enum):
    """Metric trend direction"""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


@dataclass
class PerformanceWindow:
    """Performance metrics for a time window"""

    window_start: float
    window_end: float
    latency_mean: float = 0.0
    latency_std: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput_mean: float = 0.0
    throughput_min: float = 0.0
    throughput_max: float = 0.0
    error_rate: float = 0.0
    uptime_percentage: float = 100.0
    sample_count: int = 0


@dataclass
class FailureEvent:
    """Record of a failure event"""

    timestamp: float
    duration_seconds: float
    severity: str
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    affected_requests: int = 0


@dataclass
class MinerProfile:
    """Comprehensive profile for a single miner"""

    miner_id: str
    endpoint: str
    first_seen: float
    last_updated: float

    # Current state
    current_risk: RiskLevel = RiskLevel.LOW
    latency_trend: TrendDirection = TrendDirection.STABLE
    throughput_trend: TrendDirection = TrendDirection.STABLE

    # Historical performance windows
    hourly_windows: deque = field(default_factory=lambda: deque(maxlen=24))
    daily_windows: deque = field(default_factory=lambda: deque(maxlen=30))

    # Failure history
    failure_events: List[FailureEvent] = field(default_factory=list)
    total_failures: int = 0
    mean_time_between_failures: float = 0.0
    mean_time_to_recovery: float = 0.0

    # Reliability metrics
    overall_uptime: float = 100.0
    reliability_score: float = 100.0
    predictability_score: float = 100.0

    # Behavioral patterns
    peak_hours: List[int] = field(default_factory=list)
    degradation_patterns: List[str] = field(default_factory=list)

    # Raw metrics buffer for analysis
    raw_latencies: deque = field(default_factory=lambda: deque(maxlen=10000))
    raw_throughputs: deque = field(default_factory=lambda: deque(maxlen=10000))
    raw_errors: deque = field(default_factory=lambda: deque(maxlen=10000))
    raw_timestamps: deque = field(default_factory=lambda: deque(maxlen=10000))


class MinerProfiler:
    """
    Builds and maintains comprehensive profiles of miner behavior.
    Used for long-term trend analysis and failure prediction.
    """

    def __init__(self):
        self.profiles: Dict[str, MinerProfile] = {}
        self.window_duration = 3600  # 1 hour windows

    def register_miner(self, miner_id: str, endpoint: str) -> MinerProfile:
        """Register a new miner and create its profile"""
        if miner_id not in self.profiles:
            self.profiles[miner_id] = MinerProfile(
                miner_id=miner_id,
                endpoint=endpoint,
                first_seen=time.time(),
                last_updated=time.time(),
            )
            logger.info(f"Created profile for miner: {miner_id}")
        return self.profiles[miner_id]

    def record_metric(
        self,
        miner_id: str,
        latency_ms: float,
        throughput_rps: float,
        error_rate: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a metric data point for a miner"""
        if miner_id not in self.profiles:
            logger.warning(f"Unknown miner: {miner_id}")
            return

        profile = self.profiles[miner_id]
        ts = timestamp or time.time()

        # Add to raw buffers
        profile.raw_latencies.append(latency_ms)
        profile.raw_throughputs.append(throughput_rps)
        profile.raw_errors.append(error_rate)
        profile.raw_timestamps.append(ts)
        profile.last_updated = ts

        # Update trends
        self._update_trends(profile)
        self._update_risk_level(profile)

    def record_failure(
        self,
        miner_id: str,
        duration_seconds: float,
        severity: str,
        root_cause: Optional[str] = None,
        affected_requests: int = 0,
    ) -> None:
        """Record a failure event for a miner"""
        if miner_id not in self.profiles:
            return

        profile = self.profiles[miner_id]

        event = FailureEvent(
            timestamp=time.time(),
            duration_seconds=duration_seconds,
            severity=severity,
            root_cause=root_cause,
            affected_requests=affected_requests,
        )

        profile.failure_events.append(event)
        profile.total_failures += 1

        # Update MTBF if we have multiple failures
        if len(profile.failure_events) >= 2:
            intervals = []
            for i in range(1, len(profile.failure_events)):
                interval = (
                    profile.failure_events[i].timestamp
                    - profile.failure_events[i - 1].timestamp
                )
                intervals.append(interval)
            profile.mean_time_between_failures = statistics.mean(intervals)

        # Update MTTR
        durations = [e.duration_seconds for e in profile.failure_events]
        profile.mean_time_to_recovery = statistics.mean(durations)

        # Update reliability score
        self._update_reliability_score(profile)

        logger.info(
            f"Recorded failure for {miner_id}: severity={severity}, duration={duration_seconds}s"
        )

    def _update_trends(self, profile: MinerProfile) -> None:
        """Update trend analysis for a profile"""
        if len(profile.raw_latencies) < 10:
            return

        # Analyze latency trend
        recent = list(profile.raw_latencies)[-50:]
        older = (
            list(profile.raw_latencies)[-100:-50]
            if len(profile.raw_latencies) >= 100
            else []
        )

        if older:
            recent_mean = statistics.mean(recent)
            older_mean = statistics.mean(older)

            change_pct = (
                ((recent_mean - older_mean) / older_mean) * 100 if older_mean > 0 else 0
            )

            if change_pct > 20:
                profile.latency_trend = TrendDirection.DEGRADING
            elif change_pct < -20:
                profile.latency_trend = TrendDirection.IMPROVING
            else:
                # Check volatility
                recent_std = statistics.stdev(recent) if len(recent) > 1 else 0
                if recent_std > recent_mean * 0.5:
                    profile.latency_trend = TrendDirection.VOLATILE
                else:
                    profile.latency_trend = TrendDirection.STABLE

        # Analyze throughput trend
        recent_tp = list(profile.raw_throughputs)[-50:]
        older_tp = (
            list(profile.raw_throughputs)[-100:-50]
            if len(profile.raw_throughputs) >= 100
            else []
        )

        if older_tp:
            recent_mean = statistics.mean(recent_tp)
            older_mean = statistics.mean(older_tp)

            change_pct = (
                ((recent_mean - older_mean) / older_mean) * 100 if older_mean > 0 else 0
            )

            if change_pct < -20:
                profile.throughput_trend = TrendDirection.DEGRADING
            elif change_pct > 20:
                profile.throughput_trend = TrendDirection.IMPROVING
            else:
                profile.throughput_trend = TrendDirection.STABLE

    def _update_risk_level(self, profile: MinerProfile) -> None:
        """Update risk level based on current metrics and trends"""
        risk_score = 0

        # Factor in trends
        if profile.latency_trend == TrendDirection.DEGRADING:
            risk_score += 2
        elif profile.latency_trend == TrendDirection.VOLATILE:
            risk_score += 1

        if profile.throughput_trend == TrendDirection.DEGRADING:
            risk_score += 2

        # Factor in recent error rate
        if profile.raw_errors:
            recent_errors = list(profile.raw_errors)[-20:]
            avg_error = statistics.mean(recent_errors)
            if avg_error > 5:
                risk_score += 3
            elif avg_error > 1:
                risk_score += 1

        # Factor in failure history
        recent_failures = [
            f
            for f in profile.failure_events
            if f.timestamp > time.time() - 86400  # Last 24 hours
        ]
        if len(recent_failures) >= 3:
            risk_score += 3
        elif len(recent_failures) >= 1:
            risk_score += 1

        # Classify risk
        if risk_score >= 6:
            profile.current_risk = RiskLevel.CRITICAL
        elif risk_score >= 4:
            profile.current_risk = RiskLevel.HIGH
        elif risk_score >= 2:
            profile.current_risk = RiskLevel.MEDIUM
        else:
            profile.current_risk = RiskLevel.LOW

    def _update_reliability_score(self, profile: MinerProfile) -> None:
        """Update reliability score based on failure history"""
        if not profile.failure_events:
            profile.reliability_score = 100.0
            return

        # Calculate uptime
        total_downtime = sum(f.duration_seconds for f in profile.failure_events)
        total_time = time.time() - profile.first_seen

        if total_time > 0:
            profile.overall_uptime = ((total_time - total_downtime) / total_time) * 100

        # Calculate reliability score
        # Penalize for: number of failures, severity, recency
        base_score = 100.0

        # Penalty for total failures
        base_score -= min(30, profile.total_failures * 3)

        # Penalty for recent failures (last 24h)
        recent_failures = [
            f for f in profile.failure_events if f.timestamp > time.time() - 86400
        ]
        base_score -= min(30, len(recent_failures) * 10)

        # Penalty for severe failures
        severe_failures = [
            f for f in profile.failure_events if f.severity == "critical"
        ]
        base_score -= min(20, len(severe_failures) * 5)

        profile.reliability_score = max(0, base_score)

    def compute_window_stats(
        self, profile: MinerProfile, window_seconds: int = 3600
    ) -> PerformanceWindow:
        """Compute statistics for a time window"""
        current_time = time.time()
        window_start = current_time - window_seconds

        # Filter data within window
        indices = [
            i for i, ts in enumerate(profile.raw_timestamps) if ts >= window_start
        ]

        if not indices:
            return PerformanceWindow(
                window_start=window_start, window_end=current_time, sample_count=0
            )

        latencies = [profile.raw_latencies[i] for i in indices]
        throughputs = [profile.raw_throughputs[i] for i in indices]
        errors = [profile.raw_errors[i] for i in indices]

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return PerformanceWindow(
            window_start=window_start,
            window_end=current_time,
            latency_mean=statistics.mean(latencies),
            latency_std=statistics.stdev(latencies) if n > 1 else 0,
            latency_min=min(latencies),
            latency_max=max(latencies),
            latency_p50=sorted_latencies[int(n * 0.5)] if n > 0 else 0,
            latency_p95=sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            latency_p99=sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            throughput_mean=statistics.mean(throughputs),
            throughput_min=min(throughputs),
            throughput_max=max(throughputs),
            error_rate=statistics.mean(errors),
            sample_count=n,
        )

    def get_profile(self, miner_id: str) -> Optional[MinerProfile]:
        """Get a miner's profile"""
        return self.profiles.get(miner_id)

    def get_all_profiles(self) -> Dict[str, MinerProfile]:
        """Get all miner profiles"""
        return self.profiles.copy()

    def get_miners_by_risk(self, risk_level: RiskLevel) -> List[str]:
        """Get miners at a specific risk level"""
        return [
            miner_id
            for miner_id, profile in self.profiles.items()
            if profile.current_risk == risk_level
        ]

    def get_risk_ranking(self) -> List[Tuple[str, RiskLevel, float]]:
        """Get all miners ranked by risk (highest first)"""
        risk_order = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1,
        }

        rankings = [
            (miner_id, profile.current_risk, profile.reliability_score)
            for miner_id, profile in self.profiles.items()
        ]

        return sorted(rankings, key=lambda x: (-risk_order[x[1]], x[2]))

    def get_feature_vector(self, miner_id: str) -> Optional[Dict[str, float]]:
        """
        Extract feature vector for ML model input.
        This is what gets fed to the failure predictor.
        """
        profile = self.profiles.get(miner_id)
        if not profile or len(profile.raw_latencies) < 10:
            return None

        recent_latencies = list(profile.raw_latencies)[-100:]
        recent_throughputs = list(profile.raw_throughputs)[-100:]
        recent_errors = list(profile.raw_errors)[-100:]

        # Compute features
        features = {
            "latency_mean": statistics.mean(recent_latencies),
            "latency_std": statistics.stdev(recent_latencies)
            if len(recent_latencies) > 1
            else 0,
            "latency_min": min(recent_latencies),
            "latency_max": max(recent_latencies),
            "latency_trend": 1
            if profile.latency_trend == TrendDirection.DEGRADING
            else 0,
            "latency_volatility": 1
            if profile.latency_trend == TrendDirection.VOLATILE
            else 0,
            "throughput_mean": statistics.mean(recent_throughputs),
            "throughput_std": statistics.stdev(recent_throughputs)
            if len(recent_throughputs) > 1
            else 0,
            "throughput_change": self._compute_change_rate(recent_throughputs),
            "error_rate": statistics.mean(recent_errors),
            "error_rate_change": self._compute_change_rate(recent_errors),
            "total_failures": profile.total_failures,
            "recent_failures": len(
                [f for f in profile.failure_events if f.timestamp > time.time() - 86400]
            ),
            "mtbf": profile.mean_time_between_failures,
            "mttr": profile.mean_time_to_recovery,
            "reliability_score": profile.reliability_score,
            "uptime": profile.overall_uptime,
            "risk_level": {
                RiskLevel.LOW: 0,
                RiskLevel.MEDIUM: 1,
                RiskLevel.HIGH: 2,
                RiskLevel.CRITICAL: 3,
            }[profile.current_risk],
            "node_age_hours": (time.time() - profile.first_seen) / 3600,
        }

        return features

    def _compute_change_rate(self, values: List[float]) -> float:
        """Compute rate of change (slope) for a list of values"""
        if len(values) < 10:
            return 0.0

        recent = values[-10:]
        older = values[-20:-10] if len(values) >= 20 else values[: len(values) // 2]

        if not older:
            return 0.0

        recent_mean = statistics.mean(recent)
        older_mean = statistics.mean(older)

        if older_mean == 0:
            return 0.0

        return (recent_mean - older_mean) / older_mean

    def export_profile(self, miner_id: str) -> Optional[Dict[str, Any]]:
        """Export a profile as JSON-serializable dict"""
        profile = self.profiles.get(miner_id)
        if not profile:
            return None

        return {
            "miner_id": profile.miner_id,
            "endpoint": profile.endpoint,
            "first_seen": profile.first_seen,
            "last_updated": profile.last_updated,
            "current_risk": profile.current_risk.value,
            "latency_trend": profile.latency_trend.value,
            "throughput_trend": profile.throughput_trend.value,
            "total_failures": profile.total_failures,
            "mtbf": profile.mean_time_between_failures,
            "mttr": profile.mean_time_to_recovery,
            "overall_uptime": profile.overall_uptime,
            "reliability_score": profile.reliability_score,
            "predictability_score": profile.predictability_score,
            "peak_hours": profile.peak_hours,
            "degradation_patterns": profile.degradation_patterns,
            "failure_count_24h": len(
                [f for f in profile.failure_events if f.timestamp > time.time() - 86400]
            ),
            "data_points": len(profile.raw_latencies),
        }

    def export_all_profiles(self) -> Dict[str, Any]:
        """Export all profiles"""
        return {
            miner_id: self.export_profile(miner_id) for miner_id in self.profiles.keys()
        }


# Singleton instance
_profiler_instance: Optional[MinerProfiler] = None


def get_miner_profiler() -> MinerProfiler:
    """Get or create the global miner profiler instance"""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = MinerProfiler()
    return _profiler_instance