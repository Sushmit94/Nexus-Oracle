"""
Router Service Module
Off-chain load balancer and reverse proxy for miner traffic
"""

import asyncio
import time
import logging
import json
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from aiohttp import web
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinerStatus(Enum):
    """Miner routing status"""

    ACTIVE = "active"
    DEGRADED = "degraded"
    DRAINING = "draining"
    INACTIVE = "inactive"
    BLACKLISTED = "blacklisted"


class RoutingStrategy(Enum):
    """Load balancing strategies"""

    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    WEIGHTED_RANDOM = "weighted_random"
    LEAST_CONNECTIONS = "least_connections"
    LOWEST_LATENCY = "lowest_latency"
    ADAPTIVE = "adaptive"


@dataclass
class MinerRoute:
    """Routing configuration for a single miner"""

    miner_id: str
    endpoint: str
    weight: float = 1.0
    status: MinerStatus = MinerStatus.ACTIVE
    health_score: float = 100.0
    failure_probability: float = 0.0
    current_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    average_latency: float = 0.0
    last_used: float = 0.0
    last_health_check: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RouterService:
    """
    Off-chain load balancer that routes requests to miners.
    Dynamically adjusts routing weights based on AI predictions.
    """

    def __init__(
        self,
        config_path: str = "config/thresholds.yaml",
        routing_table_path: str = "router/routing_table.json",
    ):
        self._load_config(config_path)
        self.routing_table_path = routing_table_path
        self.miners: Dict[str, MinerRoute] = {}
        self.strategy = RoutingStrategy.WEIGHTED_ROUND_ROBIN
        self._round_robin_index = 0
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._callbacks: List[callable] = []
        self._lock = asyncio.Lock()

        # Load existing routing table
        self._load_routing_table()

    def _load_config(self, config_path: str):
        """Load router configuration"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                self.config = config.get("routing", {})
        except FileNotFoundError:
            logger.warning("Config not found, using defaults")
            self.config = {}

        self.min_weight = self.config.get("min_weight", 0.05)
        self.max_weight = self.config.get("max_weight", 1.0)
        self.decay_rate = self.config.get("decay_rate", 0.1)
        self.recovery_rate = self.config.get("recovery_rate", 0.05)

    def _load_routing_table(self):
        """Load routing table from JSON file"""
        try:
            with open(self.routing_table_path, "r") as f:
                data = json.load(f)

            for miner_id, miner_data in data.get("miners", {}).items():
                self.miners[miner_id] = MinerRoute(
                    miner_id=miner_id,
                    endpoint=miner_data.get("endpoint", ""),
                    weight=miner_data.get("weight", 1.0),
                    status=MinerStatus(miner_data.get("status", "active")),
                    health_score=miner_data.get("health_score", 100),
                    metadata=miner_data.get("metadata", {}),
                )

            strategy = data.get("routing_strategy", "weighted_round_robin")
            self.strategy = RoutingStrategy(strategy)

            logger.info(f"Loaded {len(self.miners)} miners from routing table")
        except FileNotFoundError:
            logger.info("No existing routing table, starting fresh")
        except Exception as e:
            logger.error(f"Error loading routing table: {e}")

    def _save_routing_table(self):
        """Save routing table to JSON file"""
        data = {
            "version": "1.0.0",
            "last_updated": time.time(),
            "routing_strategy": self.strategy.value,
            "miners": {},
        }

        for miner_id, miner in self.miners.items():
            data["miners"][miner_id] = {
                "endpoint": miner.endpoint,
                "weight": miner.weight,
                "status": miner.status.value,
                "health_score": miner.health_score,
                "failure_probability": miner.failure_probability,
                "total_requests": miner.total_requests,
                "average_latency": miner.average_latency,
                "last_health_check": miner.last_health_check,
                "metadata": miner.metadata,
            }

        try:
            with open(self.routing_table_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving routing table: {e}")

    def register_miner(
        self,
        miner_id: str,
        endpoint: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MinerRoute:
        """Register a new miner for routing"""
        if miner_id not in self.miners:
            self.miners[miner_id] = MinerRoute(
                miner_id=miner_id,
                endpoint=endpoint,
                weight=min(self.max_weight, max(self.min_weight, weight)),
                metadata=metadata or {},
            )
            self._save_routing_table()
            logger.info(f"Registered miner: {miner_id} at {endpoint}")

        return self.miners[miner_id]

    def unregister_miner(self, miner_id: str) -> None:
        """Remove a miner from routing"""
        if miner_id in self.miners:
            del self.miners[miner_id]
            self._save_routing_table()
            logger.info(f"Unregistered miner: {miner_id}")

    async def update_weight(
        self, miner_id: str, new_weight: float, reason: str = ""
    ) -> None:
        """Update routing weight for a miner"""
        async with self._lock:
            if miner_id not in self.miners:
                return

            miner = self.miners[miner_id]
            old_weight = miner.weight
            miner.weight = min(self.max_weight, max(self.min_weight, new_weight))

            # Update status based on weight
            if miner.weight <= self.min_weight:
                miner.status = MinerStatus.DRAINING
            elif miner.weight < 0.5:
                miner.status = MinerStatus.DEGRADED
            else:
                miner.status = MinerStatus.ACTIVE

            self._save_routing_table()

            logger.info(
                f"Updated weight for {miner_id}: {old_weight:.2f} -> {miner.weight:.2f} "
                f"({reason})"
            )

            # Notify callbacks
            await self._notify_callbacks(
                {
                    "event": "weight_update",
                    "miner_id": miner_id,
                    "old_weight": old_weight,
                    "new_weight": miner.weight,
                    "reason": reason,
                }
            )

    async def update_from_arbitration(
        self,
        miner_id: str,
        failure_probability: float,
        recommended_weight: float,
        urgency: str,
    ) -> None:
        """Update routing based on arbitration result"""
        if miner_id not in self.miners:
            return

        miner = self.miners[miner_id]
        miner.failure_probability = failure_probability

        # Adjust weight based on urgency
        if urgency == "critical":
            # Immediate action
            await self.update_weight(
                miner_id, self.min_weight, "Critical risk - emergency reroute"
            )
        elif urgency == "high":
            await self.update_weight(miner_id, recommended_weight * 0.5, "High risk")
        elif urgency == "medium":
            await self.update_weight(miner_id, recommended_weight, "Medium risk")
        else:
            # Gradual recovery
            current = miner.weight
            target = min(self.max_weight, current + self.recovery_rate)
            await self.update_weight(miner_id, target, "Recovery")

    def set_miner_status(self, miner_id: str, status: MinerStatus) -> None:
        """Set miner status"""
        if miner_id in self.miners:
            self.miners[miner_id].status = status
            self._save_routing_table()

    def blacklist_miner(self, miner_id: str, reason: str = "") -> None:
        """Blacklist a miner (exclude from routing)"""
        if miner_id in self.miners:
            self.miners[miner_id].status = MinerStatus.BLACKLISTED
            self.miners[miner_id].weight = 0.0
            self._save_routing_table()
            logger.warning(f"Blacklisted miner {miner_id}: {reason}")

    def _get_active_miners(self) -> List[MinerRoute]:
        """Get list of active miners eligible for routing"""
        return [
            miner
            for miner in self.miners.values()
            if miner.status in [MinerStatus.ACTIVE, MinerStatus.DEGRADED]
            and miner.weight > 0
        ]

    def _select_weighted_round_robin(
        self, miners: List[MinerRoute]
    ) -> Optional[MinerRoute]:
        """Select miner using weighted round-robin"""
        if not miners:
            return None

        # Create weighted list
        weighted_list = []
        for miner in miners:
            # Add miner proportional to weight
            count = max(1, int(miner.weight * 10))
            weighted_list.extend([miner] * count)

        if not weighted_list:
            return miners[0]

        self._round_robin_index = (self._round_robin_index + 1) % len(weighted_list)
        return weighted_list[self._round_robin_index]

    def _select_weighted_random(self, miners: List[MinerRoute]) -> Optional[MinerRoute]:
        """Select miner using weighted random selection"""
        if not miners:
            return None

        total_weight = sum(m.weight for m in miners)
        if total_weight == 0:
            return random.choice(miners)

        r = random.uniform(0, total_weight)
        cumulative = 0

        for miner in miners:
            cumulative += miner.weight
            if r <= cumulative:
                return miner

        return miners[-1]

    def _select_least_connections(
        self, miners: List[MinerRoute]
    ) -> Optional[MinerRoute]:
        """Select miner with fewest active connections"""
        if not miners:
            return None

        # Weight by inverse of connections
        return min(miners, key=lambda m: m.current_connections / max(0.1, m.weight))

    def _select_lowest_latency(self, miners: List[MinerRoute]) -> Optional[MinerRoute]:
        """Select miner with lowest average latency"""
        if not miners:
            return None

        # Filter miners with latency data
        with_latency = [m for m in miners if m.average_latency > 0]

        if not with_latency:
            return self._select_weighted_random(miners)

        # Weight by inverse of latency
        return min(with_latency, key=lambda m: m.average_latency / max(0.1, m.weight))

    def select_miner(self) -> Optional[MinerRoute]:
        """Select a miner based on current routing strategy"""
        miners = self._get_active_miners()

        if not miners:
            logger.warning("No active miners available for routing")
            return None

        if self.strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._select_weighted_round_robin(miners)
        elif self.strategy == RoutingStrategy.WEIGHTED_RANDOM:
            return self._select_weighted_random(miners)
        elif self.strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(miners)
        elif self.strategy == RoutingStrategy.LOWEST_LATENCY:
            return self._select_lowest_latency(miners)
        elif self.strategy == RoutingStrategy.ADAPTIVE:
            # Adaptive: use different strategies based on conditions
            if any(m.failure_probability > 0.5 for m in miners):
                return self._select_weighted_random(miners)
            else:
                return self._select_weighted_round_robin(miners)

        return self._select_weighted_random(miners)

    async def route_request(
        self, request_data: Any, timeout: float = 30.0
    ) -> Tuple[Optional[Any], Optional[str], float]:
        """
        Route a request to a selected miner.

        Args:
            request_data: The request payload to forward
            timeout: Request timeout in seconds

        Returns:
            Tuple of (response_data, miner_id, latency_ms)
        """
        miner = self.select_miner()

        if not miner:
            return None, None, 0.0

        start_time = time.time()
        miner.current_connections += 1
        miner.total_requests += 1

        try:
            if self._http_session is None:
                self._http_session = aiohttp.ClientSession()

            async with self._http_session.post(
                f"{miner.endpoint}/inference",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                latency_ms = (time.time() - start_time) * 1000

                # Update miner stats
                miner.last_used = time.time()
                miner.average_latency = (
                    miner.average_latency * (miner.total_requests - 1) + latency_ms
                ) / miner.total_requests

                if response.status == 200:
                    data = await response.json()
                    return data, miner.miner_id, latency_ms
                else:
                    miner.total_errors += 1
                    return None, miner.miner_id, latency_ms

        except asyncio.TimeoutError:
            miner.total_errors += 1
            latency_ms = timeout * 1000
            logger.warning(f"Request to {miner.miner_id} timed out")
            return None, miner.miner_id, latency_ms
        except Exception as e:
            miner.total_errors += 1
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Request to {miner.miner_id} failed: {e}")
            return None, miner.miner_id, latency_ms
        finally:
            miner.current_connections -= 1

    def on_route_change(self, callback: callable) -> None:
        """Register callback for routing changes"""
        self._callbacks.append(callback)

    async def _notify_callbacks(self, event: Dict[str, Any]):
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_routing_table(self) -> Dict[str, Any]:
        """Get current routing table as dict"""
        return {
            miner_id: {
                "endpoint": miner.endpoint,
                "weight": miner.weight,
                "status": miner.status.value,
                "health_score": miner.health_score,
                "failure_probability": miner.failure_probability,
                "current_connections": miner.current_connections,
                "total_requests": miner.total_requests,
                "error_rate": (
                    miner.total_errors / miner.total_requests * 100
                    if miner.total_requests > 0
                    else 0
                ),
                "average_latency": miner.average_latency,
            }
            for miner_id, miner in self.miners.items()
        }

    def get_miner_weights(self) -> Dict[str, float]:
        """Get current weights for all miners"""
        return {miner_id: miner.weight for miner_id, miner in self.miners.items()}

    async def close(self):
        """Close HTTP session"""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()


# Singleton instance
_router_instance: Optional[RouterService] = None


def get_router_service() -> RouterService:
    """Get or create the global router service instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = RouterService()
    return _router_instance