"""
Traffic Controller Module
Orchestrates the prediction-to-routing pipeline
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ..metrics.collector import MetricsCollector, get_collector
from ..metrics.heartbeat_monitor import HeartbeatMonitor, get_heartbeat_monitor
from ..metrics.miner_profiler import MinerProfiler, get_miner_profiler
from ..agents.arbitration_nexus import (
    ArbitrationNexus,
    get_arbitration_nexus,
    ArbitrationResult,
)
from .router_service import RouterService, get_router_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ControllerMode(Enum):
    """Traffic controller operating modes"""

    AUTOMATIC = "automatic"  # Fully automated decisions
    SUPERVISED = "supervised"  # Requires approval for major changes
    MANUAL = "manual"  # Only applies manual updates
    DRY_RUN = "dry_run"  # Logs decisions without applying


@dataclass
class RoutingDecision:
    """A routing decision made by the controller"""

    miner_id: str
    timestamp: float
    old_weight: float
    new_weight: float
    reason: str
    arbitration_result: Optional[ArbitrationResult]
    applied: bool


class TrafficController:
    """
    Orchestrates the prediction-to-routing pipeline.
    Connects metrics collection -> AI prediction -> routing decisions.
    """

    def __init__(
        self,
        mode: ControllerMode = ControllerMode.AUTOMATIC,
        decision_interval: float = 10.0,
    ):
        self.mode = mode
        self.decision_interval = decision_interval
        self.running = False

        # Initialize components
        self.collector = get_collector()
        self.heartbeat_monitor = get_heartbeat_monitor()
        self.profiler = get_miner_profiler()
        self.arbitration_nexus = get_arbitration_nexus()
        self.router = get_router_service()

        # Decision tracking
        self.decisions: List[RoutingDecision] = []
        self.max_decisions = 1000
        self.pending_approvals: Dict[str, RoutingDecision] = {}

        # Callbacks
        self._decision_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []

        # Connect heartbeat alerts to controller
        self.heartbeat_monitor.on_alert(self._handle_heartbeat_alert)

    def register_miner(self, miner_id: str, endpoint: str) -> None:
        """Register a miner across all components"""
        self.collector.register_miner(miner_id, endpoint)
        self.heartbeat_monitor.register_miner(miner_id, endpoint)
        self.profiler.register_miner(miner_id, endpoint)
        self.router.register_miner(miner_id, endpoint)
        logger.info(f"Registered miner {miner_id} across all components")

    def unregister_miner(self, miner_id: str) -> None:
        """Unregister a miner from all components"""
        self.collector.unregister_miner(miner_id)
        self.heartbeat_monitor.unregister_miner(miner_id)
        # Profiler keeps history
        self.router.unregister_miner(miner_id)
        logger.info(f"Unregistered miner {miner_id}")

    def on_decision(self, callback: Callable) -> None:
        """Register callback for routing decisions"""
        self._decision_callbacks.append(callback)

    def on_alert(self, callback: Callable) -> None:
        """Register callback for alerts"""
        self._alert_callbacks.append(callback)

    async def _notify_decision(self, decision: RoutingDecision):
        """Notify callbacks of a decision"""
        for callback in self._decision_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(decision)
                else:
                    callback(decision)
            except Exception as e:
                logger.error(f"Decision callback error: {e}")

    async def _notify_alert(self, alert: Dict[str, Any]):
        """Notify callbacks of an alert"""
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def _handle_heartbeat_alert(self, alert: Dict[str, Any]):
        """Handle heartbeat alert from monitor"""
        miner_id = alert.get("miner_id")
        alert_type = alert.get("alert_type")

        logger.warning(f"Heartbeat alert for {miner_id}: {alert_type}")

        # Forward alert
        await self._notify_alert(alert)

        # Take immediate action for critical alerts
        if alert_type == "dead" and self.mode == ControllerMode.AUTOMATIC:
            await self._emergency_reroute(
                miner_id, "Node declared dead by heartbeat monitor"
            )

    async def _emergency_reroute(self, miner_id: str, reason: str):
        """Execute emergency reroute for a miner"""
        if miner_id not in self.router.miners:
            return

        old_weight = self.router.miners[miner_id].weight

        await self.router.update_weight(miner_id, 0.0, f"EMERGENCY: {reason}")

        decision = RoutingDecision(
            miner_id=miner_id,
            timestamp=time.time(),
            old_weight=old_weight,
            new_weight=0.0,
            reason=f"EMERGENCY: {reason}",
            arbitration_result=None,
            applied=True,
        )

        self._record_decision(decision)
        await self._notify_decision(decision)

        logger.critical(f"Emergency reroute executed for {miner_id}: {reason}")

    async def _process_miner(self, miner_id: str) -> Optional[RoutingDecision]:
        """Process a single miner through the prediction pipeline"""
        # Get metrics
        miner_metrics = self.collector.get_miner_metrics(miner_id)
        if not miner_metrics:
            return None

        # Get feature vector for ML
        features = self.profiler.get_feature_vector(miner_id)
        if not features:
            # Create basic features from current metrics
            features = {
                "latency_mean": miner_metrics.latency_mean,
                "latency_std": miner_metrics.latency_std,
                "latency_trend": 0,
                "latency_volatility": 0,
                "throughput_mean": miner_metrics.throughput_avg,
                "throughput_change": 0,
                "error_rate": miner_metrics.error_rate_avg,
                "error_rate_change": 0,
                "missed_heartbeats": 0,
                "heartbeat_stability": 100,
                "historical_failures": 0,
                "time_since_last_failure": 9999,
                "node_age_hours": 24,
                "risk_score": 0,
                "reliability_score": miner_metrics.health_score,
                "uptime_percentage": miner_metrics.uptime_percentage,
            }

        # Get recent latencies
        latencies = [m.latency_ms for m in list(miner_metrics.metrics_history)[-100:]]
        if not latencies:
            latencies = (
                [miner_metrics.latency_mean]
                if miner_metrics.latency_mean > 0
                else [100]
            )

        # Build metrics dict for LLM
        full_metrics = {
            **features,
            "latency_p99": miner_metrics.latency_p99,
            "recent_failures": 0,
        }

        # Run arbitration
        try:
            result = await self.arbitration_nexus.arbitrate(
                miner_id=miner_id,
                latencies=latencies,
                features=features,
                metrics=full_metrics,
            )
        except Exception as e:
            logger.error(f"Arbitration error for {miner_id}: {e}")
            return None

        # Determine if action needed
        current_weight = self.router.miners.get(miner_id)
        if not current_weight:
            return None

        old_weight = current_weight.weight
        new_weight = result.routing_weight

        # Check if change is significant
        weight_change = abs(new_weight - old_weight)
        if weight_change < 0.05:
            return None  # No significant change

        # Create decision
        decision = RoutingDecision(
            miner_id=miner_id,
            timestamp=time.time(),
            old_weight=old_weight,
            new_weight=new_weight,
            reason=result.reasoning,
            arbitration_result=result,
            applied=False,
        )

        # Apply based on mode
        if self.mode == ControllerMode.AUTOMATIC:
            await self._apply_decision(decision)
        elif self.mode == ControllerMode.SUPERVISED:
            if result.urgency in ["critical", "high"]:
                # Auto-apply urgent decisions
                await self._apply_decision(decision)
            else:
                # Queue for approval
                self.pending_approvals[miner_id] = decision
                logger.info(f"Decision for {miner_id} pending approval")
        elif self.mode == ControllerMode.DRY_RUN:
            logger.info(
                f"[DRY RUN] Would update {miner_id}: {old_weight:.2f} -> {new_weight:.2f}"
            )

        return decision

    async def _apply_decision(self, decision: RoutingDecision):
        """Apply a routing decision"""
        await self.router.update_from_arbitration(
            miner_id=decision.miner_id,
            failure_probability=decision.arbitration_result.final_failure_probability
            if decision.arbitration_result
            else 0,
            recommended_weight=decision.new_weight,
            urgency=decision.arbitration_result.urgency
            if decision.arbitration_result
            else "low",
        )

        decision.applied = True
        self._record_decision(decision)
        await self._notify_decision(decision)

    def _record_decision(self, decision: RoutingDecision):
        """Record a decision in history"""
        self.decisions.append(decision)
        if len(self.decisions) > self.max_decisions:
            self.decisions.pop(0)

    def approve_decision(self, miner_id: str) -> bool:
        """Approve a pending decision (for supervised mode)"""
        if miner_id in self.pending_approvals:
            decision = self.pending_approvals.pop(miner_id)
            asyncio.create_task(self._apply_decision(decision))
            return True
        return False

    def reject_decision(self, miner_id: str) -> bool:
        """Reject a pending decision"""
        if miner_id in self.pending_approvals:
            del self.pending_approvals[miner_id]
            return True
        return False

    async def _decision_loop(self):
        """Main decision loop"""
        while self.running:
            try:
                # Process all miners
                for miner_id in list(self.collector.miners.keys()):
                    decision = await self._process_miner(miner_id)
                    if decision:
                        logger.info(
                            f"Decision for {miner_id}: {decision.old_weight:.2f} -> "
                            f"{decision.new_weight:.2f} ({decision.reason[:50]}...)"
                        )

                await asyncio.sleep(self.decision_interval)

            except Exception as e:
                logger.error(f"Decision loop error: {e}")
                await asyncio.sleep(1)

    async def start(self):
        """Start the traffic controller"""
        self.running = True

        # Start sub-components
        asyncio.create_task(self.collector.start())
        asyncio.create_task(self.heartbeat_monitor.start_timeout_checker())

        # Start decision loop
        asyncio.create_task(self._decision_loop())

        logger.info(f"Traffic controller started in {self.mode.value} mode")

    async def stop(self):
        """Stop the traffic controller"""
        self.running = False
        await self.collector.stop()
        await self.heartbeat_monitor.stop()
        await self.router.close()
        logger.info("Traffic controller stopped")

    def set_mode(self, mode: ControllerMode):
        """Change operating mode"""
        old_mode = self.mode
        self.mode = mode
        logger.info(f"Controller mode changed: {old_mode.value} -> {mode.value}")

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status"""
        return {
            "mode": self.mode.value,
            "running": self.running,
            "registered_miners": len(self.collector.miners),
            "healthy_miners": len(self.collector.get_healthy_miners()),
            "unhealthy_miners": len(self.collector.get_unhealthy_miners()),
            "pending_approvals": len(self.pending_approvals),
            "total_decisions": len(self.decisions),
            "routing_table": self.router.get_routing_table(),
        }

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent routing decisions"""
        recent = self.decisions[-limit:]
        return [
            {
                "miner_id": d.miner_id,
                "timestamp": d.timestamp,
                "old_weight": d.old_weight,
                "new_weight": d.new_weight,
                "reason": d.reason,
                "applied": d.applied,
            }
            for d in reversed(recent)
        ]

    def get_pending_approvals(self) -> Dict[str, Dict[str, Any]]:
        """Get pending approval decisions"""
        return {
            miner_id: {
                "old_weight": d.old_weight,
                "new_weight": d.new_weight,
                "reason": d.reason,
                "timestamp": d.timestamp,
            }
            for miner_id, d in self.pending_approvals.items()
        }


# Singleton instance
_controller_instance: Optional[TrafficController] = None


def get_traffic_controller() -> TrafficController:
    """Get or create the global traffic controller instance"""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = TrafficController()
    return _controller_instance