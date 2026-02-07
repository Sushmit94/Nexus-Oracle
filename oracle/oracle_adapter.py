"""
Oracle Adapter Module
Adapts between the prediction system and various oracle interfaces
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ..router.traffic_controller import (
    TrafficController,
    get_traffic_controller,
    RoutingDecision,
)
from ..agents.arbitration_nexus import ArbitrationResult
from .oracle_publisher import OraclePublisher, get_oracle_publisher, PublishTarget

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PublishMode(Enum):
    """Oracle publishing modes"""

    IMMEDIATE = "immediate"  # Publish every decision
    BATCHED = "batched"  # Batch updates periodically
    THRESHOLD = "threshold"  # Publish only significant changes
    MANUAL = "manual"  # Manual publishing only


@dataclass
class MinerMapping:
    """Maps internal miner ID to blockchain address"""

    miner_id: str
    ethereum_address: str
    registered_at: float
    last_published: float = 0.0
    last_health_score: float = 100.0
    last_failure_prob: float = 0.0
    last_routing_weight: float = 1.0


class OracleAdapter:
    """
    Adapts between the Predictive Router Oracle system and
    on-chain/off-chain oracle interfaces.
    """

    def __init__(
        self,
        publish_mode: PublishMode = PublishMode.THRESHOLD,
        batch_interval: float = 60.0,
        change_threshold: float = 0.1,
    ):
        self.publish_mode = publish_mode
        self.batch_interval = batch_interval
        self.change_threshold = change_threshold

        # Components
        self.controller = get_traffic_controller()
        self.publisher = get_oracle_publisher()

        # Miner address mapping
        self.miner_mapping: Dict[str, MinerMapping] = {}

        # Pending batch updates
        self._pending_batch: Dict[str, Dict[str, Any]] = {}
        self._batch_lock = asyncio.Lock()

        # Publishing state
        self.running = False
        self._last_batch_time = 0.0

        # Callbacks
        self._publish_callbacks: List[Callable] = []

        # Register for routing decisions
        self.controller.on_decision(self._handle_routing_decision)

    def register_miner_address(self, miner_id: str, ethereum_address: str) -> None:
        """Register mapping between miner ID and Ethereum address"""
        self.miner_mapping[miner_id] = MinerMapping(
            miner_id=miner_id,
            ethereum_address=ethereum_address,
            registered_at=time.time(),
        )
        logger.info(f"Registered address mapping: {miner_id} -> {ethereum_address}")

    def get_ethereum_address(self, miner_id: str) -> Optional[str]:
        """Get Ethereum address for a miner ID"""
        mapping = self.miner_mapping.get(miner_id)
        return mapping.ethereum_address if mapping else None

    def on_publish(self, callback: Callable) -> None:
        """Register callback for publish events"""
        self._publish_callbacks.append(callback)

    async def _notify_publish(self, event: Dict[str, Any]):
        """Notify callbacks of publish event"""
        for callback in self._publish_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Publish callback error: {e}")

    async def _handle_routing_decision(self, decision: RoutingDecision):
        """Handle routing decision from traffic controller"""
        miner_id = decision.miner_id

        # Check if we have address mapping
        if miner_id not in self.miner_mapping:
            logger.debug(f"No address mapping for {miner_id}, skipping oracle publish")
            return

        mapping = self.miner_mapping[miner_id]

        # Prepare update data
        update_data = self._prepare_update_data(decision, mapping)

        if self.publish_mode == PublishMode.IMMEDIATE:
            await self._publish_immediate(miner_id, update_data)
        elif self.publish_mode == PublishMode.BATCHED:
            await self._queue_for_batch(miner_id, update_data)
        elif self.publish_mode == PublishMode.THRESHOLD:
            if self._exceeds_threshold(mapping, update_data):
                await self._publish_immediate(miner_id, update_data)
            else:
                logger.debug(f"Change below threshold for {miner_id}")

    def _prepare_update_data(
        self, decision: RoutingDecision, mapping: MinerMapping
    ) -> Dict[str, Any]:
        """Prepare update data from routing decision"""
        arb_result = decision.arbitration_result

        # Extract health score from arbitration if available
        health_score = 100.0
        failure_prob = 0.0

        if arb_result:
            failure_prob = arb_result.final_failure_probability
            health_score = (1 - failure_prob) * 100

        return {
            "miner_address": mapping.ethereum_address,
            "health_score": health_score,
            "failure_probability": failure_prob,
            "routing_weight": decision.new_weight,
            "evidence": {
                "reason": decision.reason,
                "old_weight": decision.old_weight,
                "timestamp": decision.timestamp,
                "consensus_level": arb_result.consensus_level.value
                if arb_result
                else None,
                "agreement_score": arb_result.agreement_score if arb_result else None,
            },
        }

    def _exceeds_threshold(
        self, mapping: MinerMapping, update_data: Dict[str, Any]
    ) -> bool:
        """Check if change exceeds publishing threshold"""
        weight_change = abs(update_data["routing_weight"] - mapping.last_routing_weight)
        prob_change = abs(
            update_data["failure_probability"] - mapping.last_failure_prob
        )

        return (
            weight_change >= self.change_threshold
            or prob_change >= self.change_threshold
        )

    async def _publish_immediate(self, miner_id: str, update_data: Dict[str, Any]):
        """Publish update immediately"""
        try:
            publications = await self.publisher.publish_health_update(
                miner_address=update_data["miner_address"],
                health_score=update_data["health_score"],
                failure_probability=update_data["failure_probability"],
                routing_weight=update_data["routing_weight"],
                evidence=update_data["evidence"],
            )

            # Update mapping state
            if miner_id in self.miner_mapping:
                mapping = self.miner_mapping[miner_id]
                mapping.last_published = time.time()
                mapping.last_health_score = update_data["health_score"]
                mapping.last_failure_prob = update_data["failure_probability"]
                mapping.last_routing_weight = update_data["routing_weight"]

            # Notify callbacks
            await self._notify_publish(
                {
                    "type": "immediate",
                    "miner_id": miner_id,
                    "data": update_data,
                    "publications": [
                        {"target": p.target.value, "success": p.success}
                        for p in publications
                    ],
                }
            )

            logger.info(f"Published update for {miner_id}")

        except Exception as e:
            logger.error(f"Failed to publish for {miner_id}: {e}")

    async def _queue_for_batch(self, miner_id: str, update_data: Dict[str, Any]):
        """Queue update for batch publishing"""
        async with self._batch_lock:
            self._pending_batch[miner_id] = update_data

    async def _process_batch(self):
        """Process and publish batched updates"""
        async with self._batch_lock:
            if not self._pending_batch:
                return

            batch_updates = list(self._pending_batch.values())
            self._pending_batch.clear()

        if batch_updates:
            try:
                publication = await self.publisher.batch_publish_onchain(batch_updates)

                # Update mapping states
                for update in batch_updates:
                    for miner_id, mapping in self.miner_mapping.items():
                        if mapping.ethereum_address == update["miner_address"]:
                            mapping.last_published = time.time()
                            mapping.last_health_score = update["health_score"]
                            mapping.last_failure_prob = update["failure_probability"]
                            mapping.last_routing_weight = update["routing_weight"]

                await self._notify_publish(
                    {
                        "type": "batch",
                        "count": len(batch_updates),
                        "success": publication.success,
                        "tx_hash": publication.tx_hash,
                    }
                )

                logger.info(f"Published batch of {len(batch_updates)} updates")

            except Exception as e:
                logger.error(f"Batch publish failed: {e}")

        self._last_batch_time = time.time()

    async def publish_emergency_reroute(self, miner_id: str, reason: str) -> bool:
        """Publish emergency reroute to oracle"""
        if miner_id not in self.miner_mapping:
            logger.warning(f"No address mapping for {miner_id}")
            return False

        mapping = self.miner_mapping[miner_id]

        try:
            publication = await self.publisher.publish_emergency_reroute(
                miner_address=mapping.ethereum_address, reason=reason
            )

            if publication.success:
                mapping.last_published = time.time()
                mapping.last_routing_weight = 0.0
                mapping.last_failure_prob = 1.0

                await self._notify_publish(
                    {
                        "type": "emergency",
                        "miner_id": miner_id,
                        "reason": reason,
                        "tx_hash": publication.tx_hash,
                    }
                )

                logger.warning(f"Emergency reroute published for {miner_id}")
                return True
            else:
                logger.error(
                    f"Emergency reroute failed for {miner_id}: {publication.error}"
                )
                return False

        except Exception as e:
            logger.error(f"Emergency reroute error for {miner_id}: {e}")
            return False

    async def force_publish(self, miner_id: str) -> bool:
        """Force publish current state for a miner"""
        if miner_id not in self.miner_mapping:
            return False

        mapping = self.miner_mapping[miner_id]

        # Get current state from controller
        result = self.controller.arbitration_nexus.get_latest_result(miner_id)

        if not result:
            logger.warning(f"No arbitration result for {miner_id}")
            return False

        update_data = {
            "miner_address": mapping.ethereum_address,
            "health_score": (1 - result.final_failure_probability) * 100,
            "failure_probability": result.final_failure_probability,
            "routing_weight": result.routing_weight,
            "evidence": {
                "reason": result.reasoning,
                "consensus_level": result.consensus_level.value,
                "forced": True,
            },
        }

        await self._publish_immediate(miner_id, update_data)
        return True

    async def _batch_loop(self):
        """Background loop for batch publishing"""
        while self.running:
            try:
                await asyncio.sleep(self.batch_interval)

                if self.publish_mode == PublishMode.BATCHED:
                    await self._process_batch()

            except Exception as e:
                logger.error(f"Batch loop error: {e}")

    async def start(self):
        """Start the oracle adapter"""
        self.running = True

        # Start batch processing loop if in batch mode
        if self.publish_mode == PublishMode.BATCHED:
            asyncio.create_task(self._batch_loop())

        logger.info(f"Oracle adapter started in {self.publish_mode.value} mode")

    async def stop(self):
        """Stop the oracle adapter"""
        self.running = False

        # Process any remaining batch
        if self._pending_batch:
            await self._process_batch()

        await self.publisher.close()
        logger.info("Oracle adapter stopped")

    def set_publish_mode(self, mode: PublishMode):
        """Change publishing mode"""
        old_mode = self.publish_mode
        self.publish_mode = mode
        logger.info(f"Publish mode changed: {old_mode.value} -> {mode.value}")

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status"""
        return {
            "publish_mode": self.publish_mode.value,
            "running": self.running,
            "registered_miners": len(self.miner_mapping),
            "pending_batch": len(self._pending_batch),
            "last_batch_time": self._last_batch_time,
            "publisher_webhooks": len(self.publisher.webhooks),
            "publication_history": len(self.publisher.publications),
        }

    def get_miner_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Get all miner address mappings"""
        return {
            miner_id: {
                "ethereum_address": m.ethereum_address,
                "registered_at": m.registered_at,
                "last_published": m.last_published,
                "last_health_score": m.last_health_score,
                "last_failure_prob": m.last_failure_prob,
                "last_routing_weight": m.last_routing_weight,
            }
            for miner_id, m in self.miner_mapping.items()
        }


# Singleton instance
_adapter_instance: Optional[OracleAdapter] = None


def get_oracle_adapter() -> OracleAdapter:
    """Get or create the global oracle adapter instance"""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = OracleAdapter()
    return _adapter_instance