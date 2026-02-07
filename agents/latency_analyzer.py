"""
Latency Analyzer Agent
Statistical anomaly detection for latency patterns
"""

import time
import logging
import statistics
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import yaml
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of latency anomalies"""

    NONE = "none"
    SPIKE = "spike"
    GRADUAL_INCREASE = "gradual_increase"
    HIGH_VARIANCE = "high_variance"
    BIMODAL = "bimodal"
    DEGRADATION = "degradation"


@dataclass
class LatencyAnalysis:
    """Result of latency analysis"""

    miner_id: str
    timestamp: float
    anomaly_type: AnomalyType
    anomaly_score: float  # 0-1, higher = more anomalous
    failure_probability: float  # 0-1, probability of impending failure
    confidence: float  # 0-1, confidence in the prediction
    details: Dict[str, Any]
    recommendation: str


class LatencyAnalyzer:
    """
    Statistical agent for analyzing latency patterns and detecting anomalies.
    Uses Isolation Forest for unsupervised anomaly detection.
    """

    def __init__(self, config_path: str = "config/models.yaml"):
        self._load_config(config_path)
        self.scalers: Dict[str, StandardScaler] = {}
        self.models: Dict[str, IsolationForest] = {}
        self.training_data: Dict[str, deque] = {}
        self.min_training_samples = 100

    def _load_config(self, config_path: str):
        """Load model configuration"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                self.config = config.get("statistical_model", {})
        except FileNotFoundError:
            logger.warning(f"Config not found, using defaults")
            self.config = {}

        # Model parameters
        self.n_estimators = self.config.get("parameters", {}).get("n_estimators", 100)
        self.contamination = self.config.get("parameters", {}).get("contamination", 0.1)
        self.window_size = self.config.get("window_size", 60)

    def _ensure_model(self, miner_id: str) -> None:
        """Ensure model exists for a miner"""
        if miner_id not in self.models:
            self.models[miner_id] = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )
            self.scalers[miner_id] = StandardScaler()
            self.training_data[miner_id] = deque(maxlen=5000)

    def _extract_features(self, latencies: List[float]) -> np.ndarray:
        """Extract statistical features from latency data"""
        if len(latencies) < 5:
            return np.zeros((1, 10))

        latencies = np.array(latencies)

        features = np.array(
            [
                np.mean(latencies),
                np.std(latencies),
                np.min(latencies),
                np.max(latencies),
                np.percentile(latencies, 50),
                np.percentile(latencies, 95),
                np.percentile(latencies, 99),
                np.max(latencies) - np.min(latencies),  # Range
                self._compute_trend(latencies),  # Trend
                self._compute_volatility(latencies),  # Volatility
            ]
        ).reshape(1, -1)

        return features

    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute linear trend slope"""
        if len(values) < 3:
            return 0.0

        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope

    def _compute_volatility(self, values: np.ndarray) -> float:
        """Compute rolling volatility"""
        if len(values) < 10:
            return 0.0

        returns = np.diff(values) / values[:-1]
        return np.std(returns) * 100

    def _detect_anomaly_type(
        self, latencies: List[float], anomaly_score: float
    ) -> AnomalyType:
        """Classify the type of anomaly"""
        if anomaly_score < 0.3:
            return AnomalyType.NONE

        latencies = np.array(latencies)
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)

        # Check for spike (sudden large values)
        recent = latencies[-10:] if len(latencies) >= 10 else latencies
        if np.max(recent) > mean_lat + 3 * std_lat:
            return AnomalyType.SPIKE

        # Check for gradual increase
        if len(latencies) >= 20:
            first_half = np.mean(latencies[: len(latencies) // 2])
            second_half = np.mean(latencies[len(latencies) // 2 :])
            if second_half > first_half * 1.3:
                return AnomalyType.GRADUAL_INCREASE

        # Check for high variance
        cv = std_lat / mean_lat if mean_lat > 0 else 0
        if cv > 0.5:
            return AnomalyType.HIGH_VARIANCE

        # Check for bimodal distribution
        if self._is_bimodal(latencies):
            return AnomalyType.BIMODAL

        # Default to degradation
        if anomaly_score >= 0.5:
            return AnomalyType.DEGRADATION

        return AnomalyType.NONE

    def _is_bimodal(self, values: np.ndarray) -> bool:
        """Check if distribution is bimodal using histogram"""
        if len(values) < 30:
            return False

        hist, _ = np.histogram(values, bins=10)

        # Look for two peaks
        peaks = 0
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks += 1

        return peaks >= 2

    def add_training_data(self, miner_id: str, latencies: List[float]) -> None:
        """Add latency data for model training"""
        self._ensure_model(miner_id)

        features = self._extract_features(latencies)
        self.training_data[miner_id].append(features.flatten())

        # Retrain if we have enough data
        if len(self.training_data[miner_id]) >= self.min_training_samples:
            self._train_model(miner_id)

    def _train_model(self, miner_id: str) -> None:
        """Train the anomaly detection model"""
        if miner_id not in self.training_data:
            return

        data = np.array(list(self.training_data[miner_id]))

        if len(data) < self.min_training_samples:
            return

        # Fit scaler and model
        scaled_data = self.scalers[miner_id].fit_transform(data)
        self.models[miner_id].fit(scaled_data)

        logger.info(f"Trained latency model for {miner_id} with {len(data)} samples")

    def analyze(self, miner_id: str, latencies: List[float]) -> LatencyAnalysis:
        """
        Analyze latency data and detect anomalies.

        Args:
            miner_id: Identifier of the miner
            latencies: Recent latency measurements (ms)

        Returns:
            LatencyAnalysis with anomaly detection results
        """
        self._ensure_model(miner_id)

        if len(latencies) < 5:
            return LatencyAnalysis(
                miner_id=miner_id,
                timestamp=time.time(),
                anomaly_type=AnomalyType.NONE,
                anomaly_score=0.0,
                failure_probability=0.0,
                confidence=0.0,
                details={"error": "Insufficient data"},
                recommendation="Collecting more data",
            )

        # Extract features
        features = self._extract_features(latencies)

        # Compute basic statistics
        stats = {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "trend": float(self._compute_trend(np.array(latencies))),
            "volatility": float(self._compute_volatility(np.array(latencies))),
        }

        # Check if model is trained
        if len(self.training_data[miner_id]) < self.min_training_samples:
            # Use heuristic-based analysis
            anomaly_score = self._heuristic_anomaly_score(latencies, stats)
            confidence = 0.5  # Lower confidence without trained model
        else:
            # Use trained model
            scaled_features = self.scalers[miner_id].transform(features)

            # Get anomaly score (-1 = anomaly, 1 = normal)
            model_score = self.models[miner_id].decision_function(scaled_features)[0]

            # Convert to 0-1 scale (higher = more anomalous)
            anomaly_score = max(0, min(1, (1 - model_score) / 2))
            confidence = 0.8

        # Determine anomaly type
        anomaly_type = self._detect_anomaly_type(latencies, anomaly_score)

        # Calculate failure probability
        failure_probability = self._calculate_failure_probability(
            anomaly_score, anomaly_type, stats
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            anomaly_type, failure_probability, stats
        )

        # Add training data
        self.add_training_data(miner_id, latencies)

        return LatencyAnalysis(
            miner_id=miner_id,
            timestamp=time.time(),
            anomaly_type=anomaly_type,
            anomaly_score=anomaly_score,
            failure_probability=failure_probability,
            confidence=confidence,
            details={
                "statistics": stats,
                "sample_count": len(latencies),
                "model_trained": len(self.training_data[miner_id])
                >= self.min_training_samples,
            },
            recommendation=recommendation,
        )

    def _heuristic_anomaly_score(self, latencies: List[float], stats: Dict) -> float:
        """Compute anomaly score using heuristics when model isn't trained"""
        score = 0.0

        # High mean latency
        if stats["mean"] > 500:
            score += 0.3
        elif stats["mean"] > 200:
            score += 0.15

        # High variance
        cv = stats["std"] / stats["mean"] if stats["mean"] > 0 else 0
        if cv > 0.5:
            score += 0.2

        # High P99
        if stats["p99"] > 1000:
            score += 0.2
        elif stats["p99"] > 500:
            score += 0.1

        # Positive trend (increasing latency)
        if stats["trend"] > 5:
            score += 0.2
        elif stats["trend"] > 2:
            score += 0.1

        # High volatility
        if stats["volatility"] > 50:
            score += 0.1

        return min(1.0, score)

    def _calculate_failure_probability(
        self, anomaly_score: float, anomaly_type: AnomalyType, stats: Dict
    ) -> float:
        """Calculate probability of failure based on analysis"""
        base_probability = anomaly_score * 0.5

        # Adjust based on anomaly type
        type_multipliers = {
            AnomalyType.NONE: 0.2,
            AnomalyType.SPIKE: 1.2,
            AnomalyType.GRADUAL_INCREASE: 1.5,
            AnomalyType.HIGH_VARIANCE: 1.1,
            AnomalyType.BIMODAL: 1.0,
            AnomalyType.DEGRADATION: 1.4,
        }

        probability = base_probability * type_multipliers.get(anomaly_type, 1.0)

        # Adjust for extreme values
        if stats["p99"] > 1000:
            probability += 0.2
        if stats["trend"] > 10:
            probability += 0.15

        return min(1.0, max(0.0, probability))

    def _generate_recommendation(
        self, anomaly_type: AnomalyType, failure_probability: float, stats: Dict
    ) -> str:
        """Generate human-readable recommendation"""
        if failure_probability < 0.3:
            return "No action required - latency within normal bounds"

        recommendations = {
            AnomalyType.SPIKE: "Investigate recent latency spike - may indicate temporary overload or network issue",
            AnomalyType.GRADUAL_INCREASE: "Latency gradually increasing - consider proactive rerouting before degradation worsens",
            AnomalyType.HIGH_VARIANCE: "High latency variance detected - miner behavior unpredictable, reduce traffic weight",
            AnomalyType.BIMODAL: "Bimodal latency distribution - possible intermittent issue, monitor closely",
            AnomalyType.DEGRADATION: "General performance degradation - recommend reducing routing weight",
            AnomalyType.NONE: "Continue monitoring",
        }

        base_rec = recommendations.get(anomaly_type, "Monitor closely")

        if failure_probability > 0.7:
            return f"HIGH RISK: {base_rec} - Immediate rerouting recommended"
        elif failure_probability > 0.5:
            return f"MEDIUM RISK: {base_rec} - Consider reducing traffic"
        else:
            return f"LOW RISK: {base_rec}"

    def save_model(self, miner_id: str, path: str) -> None:
        """Save trained model to disk"""
        if miner_id not in self.models:
            return

        model_data = {"model": self.models[miner_id], "scaler": self.scalers[miner_id]}

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved latency model for {miner_id} to {path}")

    def load_model(self, miner_id: str, path: str) -> bool:
        """Load model from disk"""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self.models[miner_id] = model_data["model"]
            self.scalers[miner_id] = model_data["scaler"]
            self.training_data[miner_id] = deque(maxlen=5000)

            logger.info(f"Loaded latency model for {miner_id} from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def export_analysis(self, analysis: LatencyAnalysis) -> Dict[str, Any]:
        """Export analysis as JSON-serializable dict"""
        return {
            "miner_id": analysis.miner_id,
            "timestamp": analysis.timestamp,
            "anomaly_type": analysis.anomaly_type.value,
            "anomaly_score": analysis.anomaly_score,
            "failure_probability": analysis.failure_probability,
            "confidence": analysis.confidence,
            "details": analysis.details,
            "recommendation": analysis.recommendation,
        }


# Singleton instance
_analyzer_instance: Optional[LatencyAnalyzer] = None


def get_latency_analyzer() -> LatencyAnalyzer:
    """Get or create the global latency analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = LatencyAnalyzer()
    return _analyzer_instance