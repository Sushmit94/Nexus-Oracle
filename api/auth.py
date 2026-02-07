"""
Authentication Module
API key management and access control for the Prediction API
"""

import time
import secrets
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccessTier(Enum):
    """API access tiers"""

    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class Permission(Enum):
    """API permissions"""

    READ_HEALTH = "read_health"
    READ_PREDICTIONS = "read_predictions"
    READ_ROUTING = "read_routing"
    WRITE_CONFIG = "write_config"
    ADMIN = "admin"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""

    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int


@dataclass
class ApiKey:
    """API key record"""

    key_id: str
    key_hash: str  # We store hash, not the actual key
    name: str
    tier: AccessTier
    permissions: List[Permission]
    rate_limit: RateLimitConfig
    created_at: float
    expires_at: Optional[float] = None
    is_active: bool = True
    owner_email: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Usage tracking
    total_requests: int = 0
    last_request: float = 0.0


@dataclass
class RequestRecord:
    """Record of an API request for rate limiting"""

    timestamp: float
    endpoint: str
    success: bool


class AuthManager:
    """
    Manages API authentication and rate limiting.
    """

    # Default rate limits per tier
    TIER_LIMITS = {
        AccessTier.FREE: RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=500,
            burst_limit=5,
        ),
        AccessTier.BASIC: RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_limit=20,
        ),
        AccessTier.PRO: RateLimitConfig(
            requests_per_minute=300,
            requests_per_hour=5000,
            requests_per_day=50000,
            burst_limit=50,
        ),
        AccessTier.ENTERPRISE: RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=50000,
            requests_per_day=500000,
            burst_limit=200,
        ),
    }

    # Default permissions per tier
    TIER_PERMISSIONS = {
        AccessTier.FREE: [Permission.READ_HEALTH],
        AccessTier.BASIC: [Permission.READ_HEALTH, Permission.READ_PREDICTIONS],
        AccessTier.PRO: [
            Permission.READ_HEALTH,
            Permission.READ_PREDICTIONS,
            Permission.READ_ROUTING,
        ],
        AccessTier.ENTERPRISE: [
            Permission.READ_HEALTH,
            Permission.READ_PREDICTIONS,
            Permission.READ_ROUTING,
            Permission.WRITE_CONFIG,
            Permission.ADMIN,
        ],
    }

    def __init__(self):
        self.api_keys: Dict[str, ApiKey] = {}  # key_id -> ApiKey
        self.key_hash_index: Dict[str, str] = {}  # key_hash -> key_id
        self.request_history: Dict[str, List[RequestRecord]] = {}  # key_id -> records
        self.history_retention_hours = 24

        # Create a default admin key for development
        self._create_dev_key()

    def _create_dev_key(self):
        """Create development API key"""
        dev_key = "dev_" + secrets.token_hex(16)
        self.create_api_key(
            name="Development Key", tier=AccessTier.ENTERPRISE, custom_key=dev_key
        )
        logger.info(f"Development API key created: {dev_key}")

    def _hash_key(self, key: str) -> str:
        """Hash an API key"""
        return hashlib.sha256(key.encode()).hexdigest()

    def create_api_key(
        self,
        name: str,
        tier: AccessTier = AccessTier.BASIC,
        permissions: Optional[List[Permission]] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        expires_in_days: Optional[int] = None,
        owner_email: Optional[str] = None,
        custom_key: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Create a new API key.

        Returns:
            Tuple of (key_id, raw_api_key)
            Note: The raw key is only returned once!
        """
        # Generate key
        raw_key = custom_key or f"pro_{secrets.token_hex(24)}"
        key_hash = self._hash_key(raw_key)
        key_id = f"key_{secrets.token_hex(8)}"

        # Set defaults
        if permissions is None:
            permissions = self.TIER_PERMISSIONS.get(tier, [Permission.READ_HEALTH])

        if rate_limit is None:
            rate_limit = self.TIER_LIMITS.get(tier, self.TIER_LIMITS[AccessTier.FREE])

        # Calculate expiry
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 86400)

        # Create key record
        api_key = ApiKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            tier=tier,
            permissions=permissions,
            rate_limit=rate_limit,
            created_at=time.time(),
            expires_at=expires_at,
            owner_email=owner_email,
        )

        self.api_keys[key_id] = api_key
        self.key_hash_index[key_hash] = key_id
        self.request_history[key_id] = []

        logger.info(f"Created API key: {key_id} ({tier.value})")

        return key_id, raw_key

    def validate_key(self, raw_key: str) -> Optional[ApiKey]:
        """
        Validate an API key.

        Returns:
            ApiKey if valid, None if invalid
        """
        key_hash = self._hash_key(raw_key)
        key_id = self.key_hash_index.get(key_hash)

        if not key_id:
            return None

        api_key = self.api_keys.get(key_id)
        if not api_key:
            return None

        # Check if active
        if not api_key.is_active:
            return None

        # Check expiry
        if api_key.expires_at and time.time() > api_key.expires_at:
            api_key.is_active = False
            return None

        return api_key

    def check_permission(self, api_key: ApiKey, permission: Permission) -> bool:
        """Check if API key has a specific permission"""
        return (
            permission in api_key.permissions or Permission.ADMIN in api_key.permissions
        )

    def check_rate_limit(self, api_key: ApiKey) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.

        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        key_id = api_key.key_id
        now = time.time()

        # Clean old records
        self._cleanup_history(key_id)

        history = self.request_history.get(key_id, [])
        limits = api_key.rate_limit

        # Calculate counts
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400

        requests_last_minute = sum(1 for r in history if r.timestamp > minute_ago)
        requests_last_hour = sum(1 for r in history if r.timestamp > hour_ago)
        requests_last_day = sum(1 for r in history if r.timestamp > day_ago)

        # Check burst (last 10 seconds)
        burst_window = now - 10
        requests_burst = sum(1 for r in history if r.timestamp > burst_window)

        # Build rate limit info
        rate_info = {
            "requests_per_minute": requests_last_minute,
            "limit_per_minute": limits.requests_per_minute,
            "requests_per_hour": requests_last_hour,
            "limit_per_hour": limits.requests_per_hour,
            "requests_per_day": requests_last_day,
            "limit_per_day": limits.requests_per_day,
            "burst": requests_burst,
            "burst_limit": limits.burst_limit,
        }

        # Check limits
        if requests_last_minute >= limits.requests_per_minute:
            rate_info["exceeded"] = "minute"
            rate_info["retry_after"] = 60 - (now - minute_ago)
            return False, rate_info

        if requests_last_hour >= limits.requests_per_hour:
            rate_info["exceeded"] = "hour"
            rate_info["retry_after"] = 3600 - (now - hour_ago)
            return False, rate_info

        if requests_last_day >= limits.requests_per_day:
            rate_info["exceeded"] = "day"
            rate_info["retry_after"] = 86400 - (now - day_ago)
            return False, rate_info

        if requests_burst >= limits.burst_limit:
            rate_info["exceeded"] = "burst"
            rate_info["retry_after"] = 10
            return False, rate_info

        return True, rate_info

    def record_request(
        self, api_key: ApiKey, endpoint: str, success: bool = True
    ) -> None:
        """Record an API request"""
        key_id = api_key.key_id

        if key_id not in self.request_history:
            self.request_history[key_id] = []

        self.request_history[key_id].append(
            RequestRecord(timestamp=time.time(), endpoint=endpoint, success=success)
        )

        # Update key stats
        api_key.total_requests += 1
        api_key.last_request = time.time()

    def _cleanup_history(self, key_id: str) -> None:
        """Clean up old request history"""
        if key_id not in self.request_history:
            return

        cutoff = time.time() - (self.history_retention_hours * 3600)
        self.request_history[key_id] = [
            r for r in self.request_history[key_id] if r.timestamp > cutoff
        ]

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False
            logger.info(f"Revoked API key: {key_id}")
            return True
        return False

    def delete_key(self, key_id: str) -> bool:
        """Permanently delete an API key"""
        if key_id in self.api_keys:
            api_key = self.api_keys[key_id]
            del self.key_hash_index[api_key.key_hash]
            del self.api_keys[key_id]
            if key_id in self.request_history:
                del self.request_history[key_id]
            logger.info(f"Deleted API key: {key_id}")
            return True
        return False

    def update_tier(self, key_id: str, new_tier: AccessTier) -> bool:
        """Update API key tier"""
        if key_id in self.api_keys:
            api_key = self.api_keys[key_id]
            api_key.tier = new_tier
            api_key.permissions = self.TIER_PERMISSIONS.get(
                new_tier, api_key.permissions
            )
            api_key.rate_limit = self.TIER_LIMITS.get(new_tier, api_key.rate_limit)
            logger.info(f"Updated tier for {key_id}: {new_tier.value}")
            return True
        return False

    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an API key"""
        api_key = self.api_keys.get(key_id)
        if not api_key:
            return None

        return {
            "key_id": api_key.key_id,
            "name": api_key.name,
            "tier": api_key.tier.value,
            "permissions": [p.value for p in api_key.permissions],
            "is_active": api_key.is_active,
            "created_at": api_key.created_at,
            "expires_at": api_key.expires_at,
            "total_requests": api_key.total_requests,
            "last_request": api_key.last_request,
            "rate_limits": {
                "per_minute": api_key.rate_limit.requests_per_minute,
                "per_hour": api_key.rate_limit.requests_per_hour,
                "per_day": api_key.rate_limit.requests_per_day,
            },
        }

    def list_keys(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """List all API keys"""
        keys = []
        for key_id, api_key in self.api_keys.items():
            if not include_inactive and not api_key.is_active:
                continue
            keys.append(self.get_key_info(key_id))
        return keys

    def get_usage_stats(self, key_id: str) -> Dict[str, Any]:
        """Get usage statistics for an API key"""
        api_key = self.api_keys.get(key_id)
        if not api_key:
            return {}

        history = self.request_history.get(key_id, [])
        now = time.time()

        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400

        return {
            "key_id": key_id,
            "total_requests": api_key.total_requests,
            "requests_last_minute": sum(1 for r in history if r.timestamp > minute_ago),
            "requests_last_hour": sum(1 for r in history if r.timestamp > hour_ago),
            "requests_last_day": sum(1 for r in history if r.timestamp > day_ago),
            "success_rate": (
                sum(1 for r in history if r.success) / len(history) * 100
                if history
                else 100
            ),
            "last_request": api_key.last_request,
            "endpoints_used": list(set(r.endpoint for r in history)),
        }


# Singleton instance
_auth_instance: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get or create the global auth manager instance"""
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = AuthManager()
    return _auth_instance