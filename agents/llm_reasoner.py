"""
LLM Reasoning Agent
Uses large language models to reason about miner health and failure likelihood
"""

import time
import logging
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningConfidence(Enum):
    """Confidence level in LLM reasoning"""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class LLMReasoning:
    """Result of LLM reasoning about failure"""

    miner_id: str
    timestamp: float
    failure_probability: float
    confidence: float
    reasoning: str
    recommended_action: str
    risk_factors: List[str]
    mitigating_factors: List[str]
    raw_response: Optional[str] = None


class LLMReasoner:
    """
    LLM-based agent for reasoning about miner failures.
    Provides explainable analysis using language models.
    """

    SYSTEM_PROMPT = """You are an expert infrastructure reliability engineer analyzing miner/validator node health in a blockchain network.

Your task is to analyze the provided metrics and determine:
1. The probability of this node failing within the next 5 minutes (0-1 scale)
2. Your confidence in this prediction (0-1 scale)
3. Key risk factors contributing to potential failure
4. Any mitigating factors that might prevent failure
5. Recommended action for the routing oracle

Consider the following metrics carefully:
- Latency: High or increasing latency often precedes failures
- Error rate: Elevated error rates indicate instability
- Throughput: Declining throughput may signal resource exhaustion
- Heartbeat stability: Missed heartbeats are early warning signs
- Historical failures: Past failures increase future risk
- Trends: Degrading trends are more concerning than stable high values

Respond ONLY with a valid JSON object in this exact format:
{
    "failure_probability": 0.0,
    "confidence": 0.0,
    "reasoning": "Brief explanation of your analysis",
    "recommended_action": "specific_action",
    "risk_factors": ["factor1", "factor2"],
    "mitigating_factors": ["factor1", "factor2"]
}

recommended_action must be one of: "no_action", "monitor_closely", "reduce_weight", "significant_reduction", "immediate_reroute"
"""

    def __init__(self, config_path: str = "config/models.yaml"):
        self._load_config(config_path)
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, LLMReasoning] = {}
        self._cache_ttl = 60  # seconds

    def _load_config(self, config_path: str):
        """Load LLM configuration"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                self.config = config.get("llm_agent", {})
        except FileNotFoundError:
            logger.warning(f"Config not found, using defaults")
            self.config = {}

        self.provider = self.config.get("provider", "openai")
        self.model = self.config.get("model", "gpt-4")
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 500)

        # API configuration
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

    def _format_metrics_prompt(self, miner_id: str, metrics: Dict[str, Any]) -> str:
        """Format metrics into a prompt for the LLM"""
        prompt = f"""Analyze the following metrics for miner {miner_id}:

CURRENT METRICS:
- Latency (mean): {metrics.get("latency_mean", "N/A")} ms
- Latency (std): {metrics.get("latency_std", "N/A")} ms
- Latency (p99): {metrics.get("latency_p99", "N/A")} ms
- Latency trend: {metrics.get("latency_trend", "N/A")} (positive = increasing)
- Throughput: {metrics.get("throughput_mean", "N/A")} RPS
- Throughput change: {metrics.get("throughput_change", "N/A")}%
- Error rate: {metrics.get("error_rate", "N/A")}%
- Missed heartbeats: {metrics.get("missed_heartbeats", "N/A")}
- Heartbeat stability: {metrics.get("heartbeat_stability", "N/A")}%

HISTORICAL DATA:
- Total historical failures: {metrics.get("historical_failures", "N/A")}
- Recent failures (24h): {metrics.get("recent_failures", "N/A")}
- Time since last failure: {metrics.get("time_since_last_failure", "N/A")} hours
- Node age: {metrics.get("node_age_hours", "N/A")} hours
- Overall reliability score: {metrics.get("reliability_score", "N/A")}/100
- Uptime percentage: {metrics.get("uptime_percentage", "N/A")}%

THRESHOLDS FOR REFERENCE:
- Warning latency: 200ms, Critical latency: 500ms
- Warning error rate: 1%, Critical error rate: 5%
- Heartbeat timeout: 30 seconds

Analyze these metrics and provide your failure prediction."""

        return prompt

    async def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API"""
        await self._ensure_session()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            async with self._http_session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error = await response.text()
                    logger.error(f"OpenAI API error: {response.status} - {error}")
                    return None
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return None

    async def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Call Anthropic API"""
        await self._ensure_session()

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with self._http_session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["content"][0]["text"]
                else:
                    error = await response.text()
                    logger.error(f"Anthropic API error: {response.status} - {error}")
                    return None
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return None

    def _parse_response(self, response: str, miner_id: str) -> Optional[LLMReasoning]:
        """Parse LLM response into structured reasoning"""
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                logger.error("No JSON found in response")
                return None

            json_str = response[start:end]
            data = json.loads(json_str)

            return LLMReasoning(
                miner_id=miner_id,
                timestamp=time.time(),
                failure_probability=float(data.get("failure_probability", 0.5)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                recommended_action=data.get("recommended_action", "monitor_closely"),
                risk_factors=data.get("risk_factors", []),
                mitigating_factors=data.get("mitigating_factors", []),
                raw_response=response,
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None

    def _get_cached(self, miner_id: str) -> Optional[LLMReasoning]:
        """Get cached reasoning if still valid"""
        if miner_id in self._cache:
            cached = self._cache[miner_id]
            if time.time() - cached.timestamp < self._cache_ttl:
                return cached
        return None

    async def reason(self, miner_id: str, metrics: Dict[str, Any]) -> LLMReasoning:
        """
        Reason about miner failure likelihood using LLM.

        Args:
            miner_id: Identifier of the miner
            metrics: Dictionary of miner metrics

        Returns:
            LLMReasoning with analysis results
        """
        # Check cache first
        cached = self._get_cached(miner_id)
        if cached:
            logger.debug(f"Using cached reasoning for {miner_id}")
            return cached

        # Check if API key is available
        if not self.api_key and self.provider == "openai":
            logger.warning("No OpenAI API key, using fallback reasoning")
            return self._fallback_reasoning(miner_id, metrics)

        # Format prompt
        prompt = self._format_metrics_prompt(miner_id, metrics)

        # Call appropriate API
        response = None
        if self.provider == "openai":
            response = await self._call_openai(prompt)
        elif self.provider == "anthropic":
            response = await self._call_anthropic(prompt)
        else:
            logger.warning(f"Unknown provider: {self.provider}")

        # Parse response
        if response:
            reasoning = self._parse_response(response, miner_id)
            if reasoning:
                self._cache[miner_id] = reasoning
                return reasoning

        # Fallback if API call failed
        return self._fallback_reasoning(miner_id, metrics)

    def _fallback_reasoning(
        self, miner_id: str, metrics: Dict[str, Any]
    ) -> LLMReasoning:
        """Provide heuristic-based reasoning when LLM is unavailable"""
        risk_factors = []
        mitigating_factors = []
        score = 0.0

        # Analyze latency
        latency_mean = metrics.get("latency_mean", 0)
        if latency_mean > 500:
            risk_factors.append("Critical latency levels")
            score += 0.25
        elif latency_mean > 200:
            risk_factors.append("Elevated latency")
            score += 0.1
        else:
            mitigating_factors.append("Latency within acceptable range")

        # Analyze error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > 5:
            risk_factors.append("High error rate")
            score += 0.25
        elif error_rate > 1:
            risk_factors.append("Elevated error rate")
            score += 0.1
        else:
            mitigating_factors.append("Low error rate")

        # Analyze heartbeats
        missed = metrics.get("missed_heartbeats", 0)
        if missed > 3:
            risk_factors.append("Multiple missed heartbeats")
            score += 0.2
        elif missed > 0:
            risk_factors.append("Some missed heartbeats")
            score += 0.1
        else:
            mitigating_factors.append("Stable heartbeat")

        # Analyze historical failures
        recent_failures = metrics.get("recent_failures", 0)
        if recent_failures > 2:
            risk_factors.append("Recent failure history")
            score += 0.15

        # Analyze reliability
        reliability = metrics.get("reliability_score", 100)
        if reliability > 90:
            mitigating_factors.append("High reliability score")
        elif reliability < 70:
            risk_factors.append("Low reliability score")
            score += 0.1

        # Determine action
        if score >= 0.7:
            action = "immediate_reroute"
            reasoning = "Multiple critical risk factors indicate imminent failure risk"
        elif score >= 0.5:
            action = "significant_reduction"
            reasoning = "Elevated risk factors warrant significant traffic reduction"
        elif score >= 0.3:
            action = "reduce_weight"
            reasoning = "Some risk factors present, recommend reducing traffic"
        elif score >= 0.1:
            action = "monitor_closely"
            reasoning = "Minor concerns detected, increased monitoring recommended"
        else:
            action = "no_action"
            reasoning = "Node operating within normal parameters"

        return LLMReasoning(
            miner_id=miner_id,
            timestamp=time.time(),
            failure_probability=min(1.0, score),
            confidence=0.6,  # Lower confidence for heuristic
            reasoning=reasoning,
            recommended_action=action,
            risk_factors=risk_factors,
            mitigating_factors=mitigating_factors,
            raw_response=None,
        )

    def reason_sync(self, miner_id: str, metrics: Dict[str, Any]) -> LLMReasoning:
        """Synchronous wrapper for reason()"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.reason(miner_id, metrics))
        finally:
            loop.close()

    async def close(self):
        """Close HTTP session"""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    def clear_cache(self):
        """Clear reasoning cache"""
        self._cache.clear()

    def export_reasoning(self, reasoning: LLMReasoning) -> Dict[str, Any]:
        """Export reasoning as JSON-serializable dict"""
        return {
            "miner_id": reasoning.miner_id,
            "timestamp": reasoning.timestamp,
            "failure_probability": reasoning.failure_probability,
            "confidence": reasoning.confidence,
            "reasoning": reasoning.reasoning,
            "recommended_action": reasoning.recommended_action,
            "risk_factors": reasoning.risk_factors,
            "mitigating_factors": reasoning.mitigating_factors,
        }


# Singleton instance
_reasoner_instance: Optional[LLMReasoner] = None


def get_llm_reasoner() -> LLMReasoner:
    """Get or create the global LLM reasoner instance"""
    global _reasoner_instance
    if _reasoner_instance is None:
        _reasoner_instance = LLMReasoner()
    return _reasoner_instance