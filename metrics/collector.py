"""
Metrics Collector Module
Continuously collects metrics from Cortensor miners/validators
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import aiohttp
import yaml
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric measurement"""

    timestamp: float
    latency_ms: float
    throughput_rps: float
    error_rate: float
    active_connections: int
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float


@dataclass
class MinerMetrics:
    """Aggregated metrics for a single miner"""

    miner_id: str
    endpoint: str
    last_seen: float
    is_healthy: bool = True
    metrics_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Computed aggregates
    latency_mean: float = 0.0
    latency_std: float = 0.0
    latency_p99: float = 0.0
    throughput_avg: float = 0.0
    error_rate_avg: float = 0.0
    uptime_percentage: float = 100.0
    health_score: float = 100.0


class MetricsCollector:
    """
    Central metrics collection service.
    Polls miners/validators and aggregates health data.
    """

    def __init__(self, config_path: str = "config/thresholds.yaml"):
        self.miners: Dict[str, MinerMetrics] = {}
        self.collection_interval = 5  # seconds
        self.running = False
        self._load_config(config_path)
        self._subscribers: List[callable] = []
        self._http_session: Optional[aiohttp.ClientSession] = None

    def _load_config(self, config_path: str):
        """Load threshold configuration"""
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}, using defaults")
            self.config = {
                "latency": {"warning": 200, "critical": 500},
                "error_rate": {"warning": 1.0, "critical": 5.0},
                "heartbeat": {"timeout_ms": 30000},
            }

    def register_miner(self, miner_id: str, endpoint: str) -> None:
        """Register a new miner for monitoring"""
        if miner_id not in self.miners:
            self.miners[miner_id] = MinerMetrics(
                miner_id=miner_id, endpoint=endpoint, last_seen=time.time()
            )
            logger.info(f"Registered miner: {miner_id} at {endpoint}")

    def unregister_miner(self, miner_id: str) -> None:
        """Remove a miner from monitoring"""
        if miner_id in self.miners:
            del self.miners[miner_id]
            logger.info(f"Unregistered miner: {miner_id}")

    def subscribe(self, callback: callable) -> None:
        """Subscribe to metric updates"""
        self._subscribers.append(callback)

    async def _notify_subscribers(self, miner_id: str, metrics: MinerMetrics):
        """Notify all subscribers of new metrics"""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(miner_id, metrics)
                else:
                    callback(miner_id, metrics)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")

    async def _collect_miner_metrics(
        self, miner: MinerMetrics
    ) -> Optional[MetricPoint]:
        """Collect metrics from a single miner"""
        start_time = time.time()

        try:
            async with self._http_session.get(
                f"{miner.endpoint}/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                latency_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    data = await response.json()

                    return MetricPoint(
                        timestamp=time.time(),
                        latency_ms=latency_ms,
                        throughput_rps=data.get("throughput", 0),
                        error_rate=data.get("error_rate", 0),
                        active_connections=data.get("connections", 0),
                        cpu_usage=data.get("cpu", 0),
                        memory_usage=data.get("memory", 0),
                        disk_io=data.get("disk_io", 0),
                        network_io=data.get("network_io", 0),
                    )
                else:
                    # Record failed request
                    return MetricPoint(
                        timestamp=time.time(),
                        latency_ms=latency_ms,
                        throughput_rps=0,
                        error_rate=100,
                        active_connections=0,
                        cpu_usage=0,
                        memory_usage=0,
                        disk_io=0,
                        network_io=0,
                    )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout collecting metrics from {miner.miner_id}")
            return MetricPoint(
                timestamp=time.time(),
                latency_ms=self.config["heartbeat"]["timeout_ms"],
                throughput_rps=0,
                error_rate=100,
                active_connections=0,
                cpu_usage=0,
                memory_usage=0,
                disk_io=0,
                network_io=0,
            )
        except Exception as e:
            logger.error(f"Error collecting from {miner.miner_id}: {e}")
            return None

    def _compute_aggregates(self, miner: MinerMetrics) -> None:
        """Compute aggregate statistics from metrics history"""
        if not miner.metrics_history:
            return

        recent_metrics = list(miner.metrics_history)[-100:]  # Last 100 points

        latencies = [m.latency_ms for m in recent_metrics]
        throughputs = [m.throughput_rps for m in recent_metrics]
        error_rates = [m.error_rate for m in recent_metrics]

        # Compute statistics
        import statistics

        miner.latency_mean = statistics.mean(latencies)
        miner.latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0
        miner.latency_p99 = (
            sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        )
        miner.throughput_avg = statistics.mean(throughputs)
        miner.error_rate_avg = statistics.mean(error_rates)

        # Compute health score (0-100)
        latency_score = max(0, 100 - (miner.latency_mean / 10))
        error_score = max(0, 100 - (miner.error_rate_avg * 10))
        throughput_score = min(100, miner.throughput_avg)

        weights = self.config.get("health_score", {})
        miner.health_score = (
            latency_score * weights.get("latency_weight", 0.25)
            + error_score * weights.get("error_rate_weight", 0.25)
            + throughput_score * weights.get("throughput_weight", 0.25)
            + miner.uptime_percentage * weights.get("uptime_weight", 0.25)
        )

        # Determine health status
        latency_config = self.config.get("latency", {})
        error_config = self.config.get("error_rate", {})

        miner.is_healthy = miner.latency_mean < latency_config.get(
            "critical", 500
        ) and miner.error_rate_avg < error_config.get("critical", 5.0)

    async def collect_all(self) -> Dict[str, MinerMetrics]:
        """Collect metrics from all registered miners"""
        tasks = []
        for miner_id, miner in self.miners.items():
            tasks.append(self._collect_miner_metrics(miner))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (miner_id, miner), result in zip(self.miners.items(), results):
            if isinstance(result, MetricPoint):
                miner.metrics_history.append(result)
                miner.last_seen = result.timestamp
                self._compute_aggregates(miner)
                await self._notify_subscribers(miner_id, miner)
            elif isinstance(result, Exception):
                logger.error(f"Collection error for {miner_id}: {result}")

        return self.miners

    async def start(self) -> None:
        """Start continuous metrics collection"""
        self.running = True
        self._http_session = aiohttp.ClientSession()
        logger.info("Metrics collector started")

        while self.running:
            try:
                await self.collect_all()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Collection cycle error: {e}")
                await asyncio.sleep(1)

        await self._http_session.close()

    async def stop(self) -> None:
        """Stop metrics collection"""
        self.running = False
        if self._http_session:
            await self._http_session.close()
        logger.info("Metrics collector stopped")

    def get_miner_metrics(self, miner_id: str) -> Optional[MinerMetrics]:
        """Get current metrics for a specific miner"""
        return self.miners.get(miner_id)

    def get_all_metrics(self) -> Dict[str, MinerMetrics]:
        """Get metrics for all miners"""
        return self.miners.copy()

    def get_healthy_miners(self) -> List[str]:
        """Get list of healthy miner IDs"""
        return [miner_id for miner_id, miner in self.miners.items() if miner.is_healthy]

    def get_unhealthy_miners(self) -> List[str]:
        """Get list of unhealthy miner IDs"""
        return [
            miner_id for miner_id, miner in self.miners.items() if not miner.is_healthy
        ]

    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics as JSON-serializable dict"""
        export_data = {}
        for miner_id, miner in self.miners.items():
            export_data[miner_id] = {
                "endpoint": miner.endpoint,
                "last_seen": miner.last_seen,
                "is_healthy": miner.is_healthy,
                "latency_mean": miner.latency_mean,
                "latency_std": miner.latency_std,
                "latency_p99": miner.latency_p99,
                "throughput_avg": miner.throughput_avg,
                "error_rate_avg": miner.error_rate_avg,
                "uptime_percentage": miner.uptime_percentage,
                "health_score": miner.health_score,
                "metrics_count": len(miner.metrics_history),
            }
        return export_data


# Singleton instance
_collector_instance: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get or create the global metrics collector instance"""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = MetricsCollector()
    return _collector_instance