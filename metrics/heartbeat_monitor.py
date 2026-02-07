"""
Heartbeat Monitor Module
Tracks miner heartbeats and detects missed beats as early warning signals
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeartbeatStatus(Enum):
    """Heartbeat health status"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclass
class HeartbeatRecord:
    """Single heartbeat record"""

    timestamp: float
    latency_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class MinerHeartbeat:
    """Heartbeat tracking for a single miner"""

    miner_id: str
    endpoint: str
    status: HeartbeatStatus = HeartbeatStatus.UNKNOWN
    last_heartbeat: float = 0.0
    consecutive_misses: int = 0
    consecutive_successes: int = 0
    total_beats: int = 0
    total_misses: int = 0
    average_latency: float = 0.0
    heartbeat_history: List[HeartbeatRecord] = field(default_factory=list)

    @property
    def miss_rate(self) -> float:
        """Calculate heartbeat miss rate"""
        if self.total_beats == 0:
            return 0.0
        return (self.total_misses / self.total_beats) * 100

    @property
    def uptime_score(self) -> float:
        """Calculate uptime score (0-100)"""
        if self.total_beats == 0:
            return 100.0
        return 100.0 - self.miss_rate

    @property
    def stability_score(self) -> float:
        """Calculate stability based on consecutive successes"""
        if self.consecutive_successes >= 100:
            return 100.0
        return min(100.0, self.consecutive_successes * 1.0)


class HeartbeatMonitor:
    """
    Monitors heartbeats from all registered miners.
    Detects early warning signs through missed heartbeats.
    """

    def __init__(self, config_path: str = "config/thresholds.yaml"):
        self.miners: Dict[str, MinerHeartbeat] = {}
        self._load_config(config_path)
        self.running = False
        self._alert_callbacks: List[Callable] = []
        self._status_callbacks: List[Callable] = []

    def _load_config(self, config_path: str):
        """Load heartbeat configuration"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                self.config = config.get("heartbeat", {})
        except FileNotFoundError:
            logger.warning(f"Config not found, using defaults")
            self.config = {}

        # Set defaults
        self.interval_ms = self.config.get("interval_ms", 5000)
        self.missed_warning = self.config.get("missed_warning", 2)
        self.missed_critical = self.config.get("missed_critical", 5)
        self.timeout_ms = self.config.get("timeout_ms", 30000)
        self.history_size = 100

    def register_miner(self, miner_id: str, endpoint: str) -> None:
        """Register a miner for heartbeat monitoring"""
        if miner_id not in self.miners:
            self.miners[miner_id] = MinerHeartbeat(
                miner_id=miner_id, endpoint=endpoint, last_heartbeat=time.time()
            )
            logger.info(f"Heartbeat monitor registered: {miner_id}")

    def unregister_miner(self, miner_id: str) -> None:
        """Remove miner from heartbeat monitoring"""
        if miner_id in self.miners:
            del self.miners[miner_id]
            logger.info(f"Heartbeat monitor unregistered: {miner_id}")

    def on_alert(self, callback: Callable) -> None:
        """Register callback for heartbeat alerts"""
        self._alert_callbacks.append(callback)

    def on_status_change(self, callback: Callable) -> None:
        """Register callback for status changes"""
        self._status_callbacks.append(callback)

    async def _trigger_alert(
        self, miner_id: str, alert_type: str, details: Dict[str, Any]
    ):
        """Trigger alert callbacks"""
        alert_data = {
            "miner_id": miner_id,
            "alert_type": alert_type,
            "timestamp": time.time(),
            "details": details,
        }

        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def _trigger_status_change(
        self, miner_id: str, old_status: HeartbeatStatus, new_status: HeartbeatStatus
    ):
        """Trigger status change callbacks"""
        status_data = {
            "miner_id": miner_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "timestamp": time.time(),
        }

        for callback in self._status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(status_data)
                else:
                    callback(status_data)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    def _update_status(self, miner: MinerHeartbeat) -> HeartbeatStatus:
        """Determine heartbeat status based on consecutive misses"""
        old_status = miner.status

        # Check for dead node (timeout)
        time_since_last = (time.time() - miner.last_heartbeat) * 1000
        if time_since_last > self.timeout_ms:
            miner.status = HeartbeatStatus.DEAD
        elif miner.consecutive_misses >= self.missed_critical:
            miner.status = HeartbeatStatus.CRITICAL
        elif miner.consecutive_misses >= self.missed_warning:
            miner.status = HeartbeatStatus.WARNING
        elif miner.consecutive_successes >= 3:
            miner.status = HeartbeatStatus.HEALTHY
        else:
            miner.status = HeartbeatStatus.UNKNOWN

        return old_status

    async def record_heartbeat(
        self,
        miner_id: str,
        success: bool,
        latency_ms: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a heartbeat result for a miner"""
        if miner_id not in self.miners:
            logger.warning(f"Unknown miner: {miner_id}")
            return

        miner = self.miners[miner_id]
        old_status = miner.status

        # Create record
        record = HeartbeatRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
        )

        # Update history (keep last N records)
        miner.heartbeat_history.append(record)
        if len(miner.heartbeat_history) > self.history_size:
            miner.heartbeat_history.pop(0)

        # Update counters
        miner.total_beats += 1

        if success:
            miner.last_heartbeat = time.time()
            miner.consecutive_successes += 1
            miner.consecutive_misses = 0

            # Update average latency
            successful_beats = [h for h in miner.heartbeat_history if h.success]
            if successful_beats:
                miner.average_latency = sum(
                    h.latency_ms for h in successful_beats
                ) / len(successful_beats)
        else:
            miner.consecutive_misses += 1
            miner.consecutive_successes = 0
            miner.total_misses += 1

            # Trigger alerts based on consecutive misses
            if miner.consecutive_misses == self.missed_warning:
                await self._trigger_alert(
                    miner_id,
                    "warning",
                    {
                        "consecutive_misses": miner.consecutive_misses,
                        "message": f"Miner {miner_id} missed {miner.consecutive_misses} heartbeats",
                    },
                )
            elif miner.consecutive_misses == self.missed_critical:
                await self._trigger_alert(
                    miner_id,
                    "critical",
                    {
                        "consecutive_misses": miner.consecutive_misses,
                        "message": f"Miner {miner_id} in critical state - {miner.consecutive_misses} missed heartbeats",
                    },
                )

        # Update status
        self._update_status(miner)

        # Trigger status change callback if status changed
        if miner.status != old_status:
            await self._trigger_status_change(miner_id, old_status, miner.status)

    def get_status(self, miner_id: str) -> Optional[HeartbeatStatus]:
        """Get current heartbeat status for a miner"""
        if miner_id in self.miners:
            return self.miners[miner_id].status
        return None

    def get_miner_heartbeat(self, miner_id: str) -> Optional[MinerHeartbeat]:
        """Get full heartbeat data for a miner"""
        return self.miners.get(miner_id)

    def get_all_status(self) -> Dict[str, HeartbeatStatus]:
        """Get status for all miners"""
        return {miner_id: miner.status for miner_id, miner in self.miners.items()}

    def get_unhealthy_miners(self) -> List[str]:
        """Get list of miners with warning/critical/dead status"""
        unhealthy = []
        for miner_id, miner in self.miners.items():
            if miner.status in [
                HeartbeatStatus.WARNING,
                HeartbeatStatus.CRITICAL,
                HeartbeatStatus.DEAD,
            ]:
                unhealthy.append(miner_id)
        return unhealthy

    def get_dead_miners(self) -> List[str]:
        """Get list of dead miners"""
        return [
            miner_id
            for miner_id, miner in self.miners.items()
            if miner.status == HeartbeatStatus.DEAD
        ]

    def get_critical_miners(self) -> List[str]:
        """Get list of miners in critical state"""
        return [
            miner_id
            for miner_id, miner in self.miners.items()
            if miner.status == HeartbeatStatus.CRITICAL
        ]

    def get_stability_ranking(self) -> List[tuple]:
        """Get miners ranked by stability score"""
        rankings = [
            (miner_id, miner.stability_score, miner.uptime_score)
            for miner_id, miner in self.miners.items()
        ]
        return sorted(rankings, key=lambda x: (x[1], x[2]), reverse=True)

    def export_data(self) -> Dict[str, Any]:
        """Export all heartbeat data as JSON-serializable dict"""
        export = {}
        for miner_id, miner in self.miners.items():
            export[miner_id] = {
                "endpoint": miner.endpoint,
                "status": miner.status.value,
                "last_heartbeat": miner.last_heartbeat,
                "consecutive_misses": miner.consecutive_misses,
                "consecutive_successes": miner.consecutive_successes,
                "total_beats": miner.total_beats,
                "total_misses": miner.total_misses,
                "miss_rate": miner.miss_rate,
                "average_latency": miner.average_latency,
                "uptime_score": miner.uptime_score,
                "stability_score": miner.stability_score,
            }
        return export

    async def check_timeouts(self) -> List[str]:
        """Check for timed out miners and update their status"""
        current_time = time.time()
        timed_out = []

        for miner_id, miner in self.miners.items():
            time_since_last = (current_time - miner.last_heartbeat) * 1000

            if (
                time_since_last > self.timeout_ms
                and miner.status != HeartbeatStatus.DEAD
            ):
                old_status = miner.status
                miner.status = HeartbeatStatus.DEAD
                timed_out.append(miner_id)

                await self._trigger_alert(
                    miner_id,
                    "dead",
                    {
                        "time_since_last_ms": time_since_last,
                        "message": f"Miner {miner_id} declared dead - no heartbeat for {time_since_last / 1000:.1f}s",
                    },
                )

                await self._trigger_status_change(
                    miner_id, old_status, HeartbeatStatus.DEAD
                )

        return timed_out

    async def start_timeout_checker(self, interval_seconds: float = 10.0) -> None:
        """Start background task to check for timeouts"""
        self.running = True
        logger.info("Heartbeat timeout checker started")

        while self.running:
            try:
                timed_out = await self.check_timeouts()
                if timed_out:
                    logger.warning(f"Timed out miners: {timed_out}")
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Timeout check error: {e}")
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the heartbeat monitor"""
        self.running = False
        logger.info("Heartbeat monitor stopped")


# Singleton instance
_monitor_instance: Optional[HeartbeatMonitor] = None


def get_heartbeat_monitor() -> HeartbeatMonitor:
    """Get or create the global heartbeat monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = HeartbeatMonitor()
    return _monitor_instance