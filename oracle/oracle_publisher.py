"""
Oracle Publisher Module
Publishes routing decisions and predictions to on-chain and off-chain systems
"""

import asyncio
import time
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from web3 import Web3
from web3.middleware import geth_poa_middleware
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PublishTarget(Enum):
    """Publication targets"""

    ONCHAIN = "onchain"
    OFFCHAIN = "offchain"
    IPFS = "ipfs"
    WEBHOOK = "webhook"


@dataclass
class Publication:
    """Record of a published update"""

    publication_id: str
    target: PublishTarget
    miner_id: str
    timestamp: float
    data: Dict[str, Any]
    tx_hash: Optional[str] = None
    ipfs_hash: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


class OraclePublisher:
    """
    Publishes routing decisions and health updates to various targets.
    Supports on-chain (Ethereum/L2), off-chain webhooks, and IPFS.
    """

    # Contract ABI (simplified for key functions)
    CONTRACT_ABI = [
        {
            "inputs": [
                {"name": "miner", "type": "address"},
                {"name": "healthScore", "type": "uint256"},
                {"name": "failureProbability", "type": "uint256"},
                {"name": "routingWeight", "type": "uint256"},
                {"name": "evidenceHash", "type": "bytes32"},
            ],
            "name": "updateMinerHealth",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [
                {"name": "miners", "type": "address[]"},
                {"name": "healthScores", "type": "uint256[]"},
                {"name": "failureProbabilities", "type": "uint256[]"},
                {"name": "routingWeights", "type": "uint256[]"},
            ],
            "name": "batchUpdateMiners",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [
                {"name": "miner", "type": "address"},
                {"name": "reason", "type": "string"},
            ],
            "name": "emergencyReroute",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
    ]

    def __init__(
        self,
        rpc_url: str = "http://localhost:8545",
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        ipfs_gateway: str = "https://ipfs.io",
        webhooks: Optional[List[str]] = None,
    ):
        self.rpc_url = rpc_url
        self.contract_address = contract_address
        self.private_key = private_key
        self.ipfs_gateway = ipfs_gateway
        self.webhooks = webhooks or []

        # Web3 setup
        self.w3: Optional[Web3] = None
        self.contract = None
        self.account = None

        # HTTP session
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Publication history
        self.publications: List[Publication] = []
        self.max_history = 1000

        # Batching
        self._pending_updates: Dict[str, Dict[str, Any]] = {}
        self._batch_lock = asyncio.Lock()

        # Initialize if credentials provided
        if rpc_url and private_key and contract_address:
            self._init_web3()

    def _init_web3(self):
        """Initialize Web3 connection"""
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

            if self.contract_address:
                self.contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(self.contract_address),
                    abi=self.CONTRACT_ABI,
                )

            if self.private_key:
                self.account = self.w3.eth.account.from_key(self.private_key)

            logger.info(f"Web3 initialized, connected: {self.w3.is_connected()}")
        except Exception as e:
            logger.error(f"Failed to initialize Web3: {e}")
            self.w3 = None

    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

    def _create_evidence_hash(self, data: Dict[str, Any]) -> bytes:
        """Create hash of prediction evidence"""
        json_str = json.dumps(data, sort_keys=True)
        return Web3.keccak(text=json_str)

    def _scale_to_contract(self, value: float, precision: int = 10000) -> int:
        """Scale float (0-1 or 0-100) to contract precision"""
        if value <= 1:
            return int(value * precision)
        return int(value * (precision / 100))

    async def publish_health_update(
        self,
        miner_address: str,
        health_score: float,
        failure_probability: float,
        routing_weight: float,
        evidence: Optional[Dict[str, Any]] = None,
        targets: Optional[List[PublishTarget]] = None,
    ) -> List[Publication]:
        """
        Publish a health update to specified targets.

        Args:
            miner_address: Ethereum address of the miner
            health_score: Health score (0-100)
            failure_probability: Failure probability (0-1)
            routing_weight: Routing weight (0-1)
            evidence: Optional evidence data
            targets: Publication targets (defaults to all)

        Returns:
            List of Publication records
        """
        targets = targets or [PublishTarget.ONCHAIN, PublishTarget.WEBHOOK]
        publications = []

        data = {
            "miner_address": miner_address,
            "health_score": health_score,
            "failure_probability": failure_probability,
            "routing_weight": routing_weight,
            "evidence": evidence,
            "timestamp": time.time(),
        }

        for target in targets:
            pub = await self._publish_to_target(target, miner_address, data)
            publications.append(pub)
            self._record_publication(pub)

        return publications

    async def _publish_to_target(
        self, target: PublishTarget, miner_id: str, data: Dict[str, Any]
    ) -> Publication:
        """Publish to a specific target"""
        pub_id = f"{target.value}_{miner_id}_{int(time.time())}"

        if target == PublishTarget.ONCHAIN:
            return await self._publish_onchain(pub_id, miner_id, data)
        elif target == PublishTarget.WEBHOOK:
            return await self._publish_webhook(pub_id, miner_id, data)
        elif target == PublishTarget.IPFS:
            return await self._publish_ipfs(pub_id, miner_id, data)
        else:
            return Publication(
                publication_id=pub_id,
                target=target,
                miner_id=miner_id,
                timestamp=time.time(),
                data=data,
                success=False,
                error="Unknown target",
            )

    async def _publish_onchain(
        self, pub_id: str, miner_id: str, data: Dict[str, Any]
    ) -> Publication:
        """Publish to on-chain contract"""
        if not self.w3 or not self.contract or not self.account:
            return Publication(
                publication_id=pub_id,
                target=PublishTarget.ONCHAIN,
                miner_id=miner_id,
                timestamp=time.time(),
                data=data,
                success=False,
                error="Web3 not initialized",
            )

        try:
            # Prepare transaction
            evidence_hash = self._create_evidence_hash(data.get("evidence", {}))

            tx = self.contract.functions.updateMinerHealth(
                Web3.to_checksum_address(data["miner_address"]),
                self._scale_to_contract(data["health_score"]),
                self._scale_to_contract(data["failure_probability"]),
                self._scale_to_contract(data["routing_weight"]),
                evidence_hash,
            ).build_transaction(
                {
                    "from": self.account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                    "gas": 200000,
                    "gasPrice": self.w3.eth.gas_price,
                }
            )

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            return Publication(
                publication_id=pub_id,
                target=PublishTarget.ONCHAIN,
                miner_id=miner_id,
                timestamp=time.time(),
                data=data,
                tx_hash=tx_hash.hex(),
                success=receipt["status"] == 1,
                error=None if receipt["status"] == 1 else "Transaction failed",
            )

        except Exception as e:
            logger.error(f"On-chain publish error: {e}")
            return Publication(
                publication_id=pub_id,
                target=PublishTarget.ONCHAIN,
                miner_id=miner_id,
                timestamp=time.time(),
                data=data,
                success=False,
                error=str(e),
            )

    async def _publish_webhook(
        self, pub_id: str, miner_id: str, data: Dict[str, Any]
    ) -> Publication:
        """Publish to webhooks"""
        await self._ensure_session()

        payload = {
            "publication_id": pub_id,
            "miner_id": miner_id,
            "data": data,
            "timestamp": time.time(),
        }

        success_count = 0
        errors = []

        for webhook_url in self.webhooks:
            try:
                async with self._http_session.post(
                    webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        success_count += 1
                    else:
                        errors.append(f"{webhook_url}: {response.status}")
            except Exception as e:
                errors.append(f"{webhook_url}: {str(e)}")

        return Publication(
            publication_id=pub_id,
            target=PublishTarget.WEBHOOK,
            miner_id=miner_id,
            timestamp=time.time(),
            data=data,
            success=success_count > 0,
            error="; ".join(errors) if errors else None,
        )

    async def _publish_ipfs(
        self, pub_id: str, miner_id: str, data: Dict[str, Any]
    ) -> Publication:
        """Publish to IPFS"""
        await self._ensure_session()

        try:
            # Prepare data for IPFS
            ipfs_data = {
                "publication_id": pub_id,
                "miner_id": miner_id,
                "data": data,
                "timestamp": time.time(),
                "version": "1.0.0",
            }

            # Add to IPFS (using Infura or similar)
            async with self._http_session.post(
                f"{self.ipfs_gateway}/api/v0/add",
                data={"file": json.dumps(ipfs_data)},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    ipfs_hash = result.get("Hash")

                    return Publication(
                        publication_id=pub_id,
                        target=PublishTarget.IPFS,
                        miner_id=miner_id,
                        timestamp=time.time(),
                        data=data,
                        ipfs_hash=ipfs_hash,
                        success=True,
                    )
                else:
                    return Publication(
                        publication_id=pub_id,
                        target=PublishTarget.IPFS,
                        miner_id=miner_id,
                        timestamp=time.time(),
                        data=data,
                        success=False,
                        error=f"IPFS error: {response.status}",
                    )

        except Exception as e:
            logger.error(f"IPFS publish error: {e}")
            return Publication(
                publication_id=pub_id,
                target=PublishTarget.IPFS,
                miner_id=miner_id,
                timestamp=time.time(),
                data=data,
                success=False,
                error=str(e),
            )

    async def batch_publish_onchain(self, updates: List[Dict[str, Any]]) -> Publication:
        """
        Batch publish multiple updates on-chain.
        More gas efficient than individual updates.
        """
        if not self.w3 or not self.contract or not self.account:
            return Publication(
                publication_id=f"batch_{int(time.time())}",
                target=PublishTarget.ONCHAIN,
                miner_id="batch",
                timestamp=time.time(),
                data={"updates": updates},
                success=False,
                error="Web3 not initialized",
            )

        try:
            miners = []
            health_scores = []
            failure_probs = []
            weights = []

            for update in updates:
                miners.append(Web3.to_checksum_address(update["miner_address"]))
                health_scores.append(self._scale_to_contract(update["health_score"]))
                failure_probs.append(
                    self._scale_to_contract(update["failure_probability"])
                )
                weights.append(self._scale_to_contract(update["routing_weight"]))

            tx = self.contract.functions.batchUpdateMiners(
                miners, health_scores, failure_probs, weights
            ).build_transaction(
                {
                    "from": self.account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                    "gas": 100000 + (50000 * len(updates)),
                    "gasPrice": self.w3.eth.gas_price,
                }
            )

            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            return Publication(
                publication_id=f"batch_{int(time.time())}",
                target=PublishTarget.ONCHAIN,
                miner_id="batch",
                timestamp=time.time(),
                data={"updates": updates, "count": len(updates)},
                tx_hash=tx_hash.hex(),
                success=receipt["status"] == 1,
            )

        except Exception as e:
            logger.error(f"Batch publish error: {e}")
            return Publication(
                publication_id=f"batch_{int(time.time())}",
                target=PublishTarget.ONCHAIN,
                miner_id="batch",
                timestamp=time.time(),
                data={"updates": updates},
                success=False,
                error=str(e),
            )

    async def publish_emergency_reroute(
        self, miner_address: str, reason: str
    ) -> Publication:
        """Publish emergency reroute on-chain"""
        if not self.w3 or not self.contract or not self.account:
            return Publication(
                publication_id=f"emergency_{int(time.time())}",
                target=PublishTarget.ONCHAIN,
                miner_id=miner_address,
                timestamp=time.time(),
                data={"reason": reason},
                success=False,
                error="Web3 not initialized",
            )

        try:
            tx = self.contract.functions.emergencyReroute(
                Web3.to_checksum_address(miner_address), reason
            ).build_transaction(
                {
                    "from": self.account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                    "gas": 150000,
                    "gasPrice": int(
                        self.w3.eth.gas_price * 1.5
                    ),  # Higher gas for emergency
                }
            )

            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            return Publication(
                publication_id=f"emergency_{int(time.time())}",
                target=PublishTarget.ONCHAIN,
                miner_id=miner_address,
                timestamp=time.time(),
                data={"reason": reason},
                tx_hash=tx_hash.hex(),
                success=receipt["status"] == 1,
            )

        except Exception as e:
            logger.error(f"Emergency reroute publish error: {e}")
            return Publication(
                publication_id=f"emergency_{int(time.time())}",
                target=PublishTarget.ONCHAIN,
                miner_id=miner_address,
                timestamp=time.time(),
                data={"reason": reason},
                success=False,
                error=str(e),
            )

    def _record_publication(self, pub: Publication):
        """Record publication in history"""
        self.publications.append(pub)
        if len(self.publications) > self.max_history:
            self.publications.pop(0)

    def add_webhook(self, url: str):
        """Add a webhook endpoint"""
        if url not in self.webhooks:
            self.webhooks.append(url)

    def remove_webhook(self, url: str):
        """Remove a webhook endpoint"""
        if url in self.webhooks:
            self.webhooks.remove(url)

    def get_publication_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent publication history"""
        recent = self.publications[-limit:]
        return [
            {
                "publication_id": p.publication_id,
                "target": p.target.value,
                "miner_id": p.miner_id,
                "timestamp": p.timestamp,
                "tx_hash": p.tx_hash,
                "ipfs_hash": p.ipfs_hash,
                "success": p.success,
                "error": p.error,
            }
            for p in reversed(recent)
        ]

    async def close(self):
        """Close HTTP session"""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()


# Singleton instance
_publisher_instance: Optional[OraclePublisher] = None


def get_oracle_publisher() -> OraclePublisher:
    """Get or create the global oracle publisher instance"""
    global _publisher_instance
    if _publisher_instance is None:
        _publisher_instance = OraclePublisher()
    return _publisher_instance