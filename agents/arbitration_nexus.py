"""
Agent Arbitration Nexus
Meta-agent that arbitrates between multiple prediction models
"""

import time
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import yaml

from .latency_analyzer import LatencyAnalyzer, LatencyAnalysis, get_latency_analyzer
from .failure_predictor import (
    FailurePredictor,
    FailurePrediction,
    get_failure_predictor,
)
from .llm_reasoner import LLMReasoner, LLMReasoning, get_llm_reasoner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsensusLevel(Enum):
    """Level of agreement between agents"""

    UNANIMOUS = "unanimous"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    DISAGREEMENT = "disagreement"


class RecommendedAction(Enum):
    """Possible routing actions"""

    NO_ACTION = "no_action"
    MONITOR = "monitor"
    REDUCE_WEIGHT = "reduce_weight"
    SIGNIFICANT_REDUCTION = "significant_reduction"
    REROUTE = "reroute"
    EMERGENCY_REROUTE = "emergency_reroute"


@dataclass
class AgentPrediction:
    """Individual agent prediction for arbitration"""

    agent_name: str
    failure_probability: float
    confidence: float
    weight: float
    recommendation: str
    details: Dict[str, Any]


@dataclass
class ArbitrationResult:
    """Result of the arbitration process"""

    miner_id: str
    timestamp: float

    # Consensus output
    final_failure_probability: float
    final_confidence: float
    consensus_level: ConsensusLevel
    recommended_action: RecommendedAction

    # Individual predictions
    agent_predictions: List[AgentPrediction]

    # Analysis
    agreement_score: float
    outliers_detected: List[str]
    reasoning: str

    # Routing directive
    routing_weight: float  # 0-1, suggested routing weight
    urgency: str  # low, medium, high, critical


class ArbitrationNexus:
    """
    Meta-agent that arbitrates between multiple AI prediction agents.
    Implements consensus-based decision making for failure predictions.
    """

    def __init__(self, config_path: str = "config/models.yaml"):
        self._load_config(config_path)

        # Initialize sub-agents
        self.latency_analyzer = get_latency_analyzer()
        self.failure_predictor = get_failure_predictor()
        self.llm_reasoner = get_llm_reasoner()

        # Arbitration state
        self._history: Dict[str, List[ArbitrationResult]] = {}
        self._history_size = 100

    def _load_config(self, config_path: str):
        """Load arbitration configuration"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                arb_config = config.get("arbitration_nexus", {})
                threshold_config = config.get("arbitration", {})
        except FileNotFoundError:
            logger.warning("Config not found, using defaults")
            arb_config = {}
            threshold_config = {}

        # Agent weights
        self.statistical_weight = threshold_config.get("statistical_weight", 0.3)
        self.ml_weight = threshold_config.get("ml_weight", 0.4)
        self.llm_weight = threshold_config.get("llm_weight", 0.3)

        # Consensus parameters
        self.agreement_threshold = threshold_config.get("agreement_threshold", 2)
        self.outlier_deviation = threshold_config.get("outlier_deviation", 0.3)
        self.confidence_boost = arb_config.get("consensus_rules", {}).get(
            "confidence_boost", 0.1
        )
        self.outlier_penalty = arb_config.get("consensus_rules", {}).get(
            "outlier_penalty", 0.2
        )

        # Action thresholds
        self.reroute_threshold = 0.7
        self.significant_reduction_threshold = 0.5
        self.reduce_weight_threshold = 0.3
        self.monitor_threshold = 0.15

    async def _get_statistical_prediction(
        self, miner_id: str, latencies: List[float]
    ) -> AgentPrediction:
        """Get prediction from statistical agent"""
        try:
            analysis = self.latency_analyzer.analyze(miner_id, latencies)

            return AgentPrediction(
                agent_name="statistical",
                failure_probability=analysis.failure_probability,
                confidence=analysis.confidence,
                weight=self.statistical_weight,
                recommendation=analysis.recommendation,
                details={
                    "anomaly_type": analysis.anomaly_type.value,
                    "anomaly_score": analysis.anomaly_score,
                },
            )
        except Exception as e:
            logger.error(f"Statistical agent error: {e}")
            return AgentPrediction(
                agent_name="statistical",
                failure_probability=0.5,
                confidence=0.0,
                weight=self.statistical_weight,
                recommendation="Error in statistical analysis",
                details={"error": str(e)},
            )

    async def _get_ml_prediction(
        self, miner_id: str, features: Dict[str, float]
    ) -> AgentPrediction:
        """Get prediction from ML agent"""
        try:
            prediction = self.failure_predictor.predict(miner_id, features)

            return AgentPrediction(
                agent_name="ml",
                failure_probability=prediction.failure_probability,
                confidence=prediction.confidence,
                weight=self.ml_weight,
                recommendation=prediction.recommendation,
                details={
                    "risk_level": prediction.risk_level.value,
                    "time_to_failure": prediction.time_to_failure_estimate,
                    "contributing_factors": prediction.contributing_factors,
                },
            )
        except Exception as e:
            logger.error(f"ML agent error: {e}")
            return AgentPrediction(
                agent_name="ml",
                failure_probability=0.5,
                confidence=0.0,
                weight=self.ml_weight,
                recommendation="Error in ML prediction",
                details={"error": str(e)},
            )

    async def _get_llm_prediction(
        self, miner_id: str, metrics: Dict[str, Any]
    ) -> AgentPrediction:
        """Get prediction from LLM agent"""
        try:
            reasoning = await self.llm_reasoner.reason(miner_id, metrics)

            return AgentPrediction(
                agent_name="llm",
                failure_probability=reasoning.failure_probability,
                confidence=reasoning.confidence,
                weight=self.llm_weight,
                recommendation=reasoning.recommended_action,
                details={
                    "reasoning": reasoning.reasoning,
                    "risk_factors": reasoning.risk_factors,
                    "mitigating_factors": reasoning.mitigating_factors,
                },
            )
        except Exception as e:
            logger.error(f"LLM agent error: {e}")
            return AgentPrediction(
                agent_name="llm",
                failure_probability=0.5,
                confidence=0.0,
                weight=self.llm_weight,
                recommendation="Error in LLM reasoning",
                details={"error": str(e)},
            )

    def _detect_outliers(self, predictions: List[AgentPrediction]) -> List[str]:
        """Detect outlier predictions"""
        if len(predictions) < 2:
            return []

        probabilities = [p.failure_probability for p in predictions]
        mean_prob = statistics.mean(probabilities)

        outliers = []
        for pred in predictions:
            if abs(pred.failure_probability - mean_prob) > self.outlier_deviation:
                outliers.append(pred.agent_name)

        return outliers

    def _calculate_consensus(
        self, predictions: List[AgentPrediction], outliers: List[str]
    ) -> Tuple[float, float, ConsensusLevel]:
        """Calculate consensus probability and confidence"""
        # Filter out outliers for main calculation
        valid_predictions = [p for p in predictions if p.agent_name not in outliers]

        if not valid_predictions:
            valid_predictions = predictions

        # Weighted average of probabilities
        total_weight = sum(p.weight * p.confidence for p in valid_predictions)
        if total_weight == 0:
            total_weight = 1

        weighted_prob = (
            sum(
                p.failure_probability * p.weight * p.confidence
                for p in valid_predictions
            )
            / total_weight
        )

        # Calculate agreement score
        probabilities = [p.failure_probability for p in valid_predictions]
        if len(probabilities) > 1:
            std = statistics.stdev(probabilities)
            agreement_score = max(0, 1 - std * 2)
        else:
            agreement_score = 0.5

        # Determine consensus level
        if len(outliers) == 0 and agreement_score > 0.9:
            consensus_level = ConsensusLevel.UNANIMOUS
        elif agreement_score > 0.8:
            consensus_level = ConsensusLevel.STRONG
        elif agreement_score > 0.6:
            consensus_level = ConsensusLevel.MODERATE
        elif agreement_score > 0.4:
            consensus_level = ConsensusLevel.WEAK
        else:
            consensus_level = ConsensusLevel.DISAGREEMENT

        # Calculate final confidence
        base_confidence = statistics.mean([p.confidence for p in valid_predictions])

        # Boost confidence if unanimous, penalize if disagreement
        if consensus_level == ConsensusLevel.UNANIMOUS:
            final_confidence = min(0.95, base_confidence + self.confidence_boost)
        elif consensus_level == ConsensusLevel.DISAGREEMENT:
            final_confidence = max(0.3, base_confidence - self.outlier_penalty)
        else:
            final_confidence = base_confidence

        return weighted_prob, final_confidence, consensus_level

    def _determine_action(
        self, probability: float, confidence: float, consensus: ConsensusLevel
    ) -> RecommendedAction:
        """Determine recommended action based on probability and confidence"""
        # Adjust thresholds based on consensus
        threshold_multiplier = {
            ConsensusLevel.UNANIMOUS: 0.9,
            ConsensusLevel.STRONG: 0.95,
            ConsensusLevel.MODERATE: 1.0,
            ConsensusLevel.WEAK: 1.1,
            ConsensusLevel.DISAGREEMENT: 1.2,
        }.get(consensus, 1.0)

        effective_prob = probability / threshold_multiplier

        if effective_prob >= 0.9:
            return RecommendedAction.EMERGENCY_REROUTE
        elif effective_prob >= self.reroute_threshold:
            return RecommendedAction.REROUTE
        elif effective_prob >= self.significant_reduction_threshold:
            return RecommendedAction.SIGNIFICANT_REDUCTION
        elif effective_prob >= self.reduce_weight_threshold:
            return RecommendedAction.REDUCE_WEIGHT
        elif effective_prob >= self.monitor_threshold:
            return RecommendedAction.MONITOR
        else:
            return RecommendedAction.NO_ACTION

    def _calculate_routing_weight(
        self, probability: float, action: RecommendedAction
    ) -> float:
        """Calculate suggested routing weight"""
        weight_map = {
            RecommendedAction.NO_ACTION: 1.0,
            RecommendedAction.MONITOR: 0.9,
            RecommendedAction.REDUCE_WEIGHT: 0.6,
            RecommendedAction.SIGNIFICANT_REDUCTION: 0.3,
            RecommendedAction.REROUTE: 0.1,
            RecommendedAction.EMERGENCY_REROUTE: 0.0,
        }

        base_weight = weight_map.get(action, 0.5)

        # Fine-tune based on probability
        adjustment = (1 - probability) * 0.2

        return max(0.0, min(1.0, base_weight + adjustment))

    def _determine_urgency(self, action: RecommendedAction, probability: float) -> str:
        """Determine urgency level"""
        if action in [RecommendedAction.EMERGENCY_REROUTE]:
            return "critical"
        elif action in [RecommendedAction.REROUTE] or probability > 0.8:
            return "high"
        elif action in [
            RecommendedAction.SIGNIFICANT_REDUCTION,
            RecommendedAction.REDUCE_WEIGHT,
        ]:
            return "medium"
        else:
            return "low"

    def _generate_reasoning(
        self,
        predictions: List[AgentPrediction],
        consensus: ConsensusLevel,
        outliers: List[str],
        action: RecommendedAction,
    ) -> str:
        """Generate human-readable reasoning"""
        parts = []

        # Describe consensus
        consensus_desc = {
            ConsensusLevel.UNANIMOUS: "All agents agree",
            ConsensusLevel.STRONG: "Strong agreement between agents",
            ConsensusLevel.MODERATE: "Moderate agreement between agents",
            ConsensusLevel.WEAK: "Weak agreement between agents",
            ConsensusLevel.DISAGREEMENT: "Agents disagree significantly",
        }
        parts.append(consensus_desc.get(consensus, ""))

        # Describe outliers
        if outliers:
            parts.append(f"Outlier predictions from: {', '.join(outliers)}")

        # Key factors from agents
        for pred in predictions:
            if pred.agent_name == "llm" and "risk_factors" in pred.details:
                factors = pred.details.get("risk_factors", [])[:2]
                if factors:
                    parts.append(f"Key risks: {', '.join(factors)}")
                break

        # Action reasoning
        action_desc = {
            RecommendedAction.NO_ACTION: "No action required at this time",
            RecommendedAction.MONITOR: "Increased monitoring recommended",
            RecommendedAction.REDUCE_WEIGHT: "Recommend reducing routing weight",
            RecommendedAction.SIGNIFICANT_REDUCTION: "Significant traffic reduction advised",
            RecommendedAction.REROUTE: "Traffic rerouting recommended",
            RecommendedAction.EMERGENCY_REROUTE: "EMERGENCY: Immediate rerouting required",
        }
        parts.append(action_desc.get(action, ""))

        return ". ".join(filter(None, parts)) + "."

    async def arbitrate(
        self,
        miner_id: str,
        latencies: List[float],
        features: Dict[str, float],
        metrics: Dict[str, Any],
    ) -> ArbitrationResult:
        """
        Run all prediction agents and arbitrate their outputs.

        Args:
            miner_id: Identifier of the miner
            latencies: Recent latency measurements
            features: Feature dictionary for ML model
            metrics: Full metrics dictionary for LLM

        Returns:
            ArbitrationResult with consensus prediction
        """
        # Gather predictions from all agents in parallel
        predictions = await asyncio.gather(
            self._get_statistical_prediction(miner_id, latencies),
            self._get_ml_prediction(miner_id, features),
            self._get_llm_prediction(miner_id, metrics),
        )

        predictions = list(predictions)

        # Detect outliers
        outliers = self._detect_outliers(predictions)

        # Calculate consensus
        final_prob, final_conf, consensus_level = self._calculate_consensus(
            predictions, outliers
        )

        # Calculate agreement score
        probs = [p.failure_probability for p in predictions]
        agreement_score = (
            max(0, 1 - statistics.stdev(probs) * 2) if len(probs) > 1 else 0.5
        )

        # Determine action
        action = self._determine_action(final_prob, final_conf, consensus_level)

        # Calculate routing weight
        routing_weight = self._calculate_routing_weight(final_prob, action)

        # Determine urgency
        urgency = self._determine_urgency(action, final_prob)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            predictions, consensus_level, outliers, action
        )

        result = ArbitrationResult(
            miner_id=miner_id,
            timestamp=time.time(),
            final_failure_probability=final_prob,
            final_confidence=final_conf,
            consensus_level=consensus_level,
            recommended_action=action,
            agent_predictions=predictions,
            agreement_score=agreement_score,
            outliers_detected=outliers,
            reasoning=reasoning,
            routing_weight=routing_weight,
            urgency=urgency,
        )

        # Store in history
        if miner_id not in self._history:
            self._history[miner_id] = []
        self._history[miner_id].append(result)
        if len(self._history[miner_id]) > self._history_size:
            self._history[miner_id].pop(0)

        logger.info(
            f"Arbitration for {miner_id}: prob={final_prob:.2f}, "
            f"conf={final_conf:.2f}, action={action.value}"
        )

        return result

    def arbitrate_sync(
        self,
        miner_id: str,
        latencies: List[float],
        features: Dict[str, float],
        metrics: Dict[str, Any],
    ) -> ArbitrationResult:
        """Synchronous wrapper for arbitrate()"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.arbitrate(miner_id, latencies, features, metrics)
            )
        finally:
            loop.close()

    def get_history(self, miner_id: str) -> List[ArbitrationResult]:
        """Get arbitration history for a miner"""
        return self._history.get(miner_id, [])

    def get_latest_result(self, miner_id: str) -> Optional[ArbitrationResult]:
        """Get most recent arbitration result"""
        history = self._history.get(miner_id, [])
        return history[-1] if history else None

    def export_result(self, result: ArbitrationResult) -> Dict[str, Any]:
        """Export arbitration result as JSON-serializable dict"""
        return {
            "miner_id": result.miner_id,
            "timestamp": result.timestamp,
            "final_failure_probability": result.final_failure_probability,
            "final_confidence": result.final_confidence,
            "consensus_level": result.consensus_level.value,
            "recommended_action": result.recommended_action.value,
            "agent_predictions": [
                {
                    "agent_name": p.agent_name,
                    "failure_probability": p.failure_probability,
                    "confidence": p.confidence,
                    "recommendation": p.recommendation,
                }
                for p in result.agent_predictions
            ],
            "agreement_score": result.agreement_score,
            "outliers_detected": result.outliers_detected,
            "reasoning": result.reasoning,
            "routing_weight": result.routing_weight,
            "urgency": result.urgency,
        }


# Singleton instance
_nexus_instance: Optional[ArbitrationNexus] = None


def get_arbitration_nexus() -> ArbitrationNexus:
    """Get or create the global arbitration nexus instance"""
    global _nexus_instance
    if _nexus_instance is None:
        _nexus_instance = ArbitrationNexus()
    return _nexus_instance