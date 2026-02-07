"""
Failure Predictor Agent
ML-based binary classification model for predicting node failures
"""

import time
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureRisk(Enum):
    """Failure risk classification"""

    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    IMMINENT = "imminent"


@dataclass
class FailurePrediction:
    """Result of failure prediction"""

    miner_id: str
    timestamp: float
    failure_probability: float  # 0-1
    risk_level: FailureRisk
    confidence: float  # 0-1
    time_to_failure_estimate: Optional[float]  # seconds
    contributing_factors: List[Dict[str, Any]]
    recommendation: str


class FailurePredictor:
    """
    ML-based agent for predicting miner/validator failures.
    Uses Gradient Boosting Classifier for binary classification.
    """

    FEATURE_NAMES = [
        "latency_mean",
        "latency_std",
        "latency_trend",
        "latency_volatility",
        "throughput_mean",
        "throughput_change",
        "error_rate",
        "error_rate_change",
        "missed_heartbeats",
        "heartbeat_stability",
        "historical_failures",
        "time_since_last_failure",
        "node_age_hours",
        "risk_score",
        "reliability_score",
        "uptime_percentage",
    ]

    def __init__(self, config_path: str = "config/models.yaml"):
        self._load_config(config_path)
        self.model: Optional[GradientBoostingClassifier] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history: List[Dict] = []
        self.feature_importance: Dict[str, float] = {}

        # Initialize model
        self._initialize_model()

    def _load_config(self, config_path: str):
        """Load model configuration"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                self.config = config.get("ml_model", {})
        except FileNotFoundError:
            logger.warning(f"Config not found, using defaults")
            self.config = {}

        # Model parameters
        params = self.config.get("parameters", {})
        self.n_estimators = params.get("n_estimators", 200)
        self.max_depth = params.get("max_depth", 6)
        self.learning_rate = params.get("learning_rate", 0.1)
        self.threshold = self.config.get("threshold", 0.5)

    def _initialize_model(self):
        """Initialize the ML model"""
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )

    def _validate_features(self, features: Dict[str, float]) -> np.ndarray:
        """Validate and convert feature dict to array"""
        feature_vector = []

        for name in self.FEATURE_NAMES:
            value = features.get(name, 0.0)
            if value is None or np.isnan(value) or np.isinf(value):
                value = 0.0
            feature_vector.append(value)

        return np.array(feature_vector).reshape(1, -1)

    def train(
        self, X: List[Dict[str, float]], y: List[int], test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the failure prediction model.

        Args:
            X: List of feature dictionaries
            y: List of labels (0 = no failure, 1 = failure)
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary of performance metrics
        """
        if len(X) < 50:
            logger.warning("Insufficient training data")
            return {"error": "Insufficient data"}

        # Convert to arrays
        X_array = np.vstack([self._validate_features(f) for f in X])
        y_array = np.array(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=test_size, random_state=42, stratify=y_array
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        # Store feature importance
        for name, importance in zip(
            self.FEATURE_NAMES, self.model.feature_importances_
        ):
            self.feature_importance[name] = float(importance)

        # Record training history
        self.training_history.append(
            {"timestamp": time.time(), "metrics": metrics, "samples": len(X)}
        )

        logger.info(
            f"Model trained - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}"
        )

        return metrics

    def predict(self, miner_id: str, features: Dict[str, float]) -> FailurePrediction:
        """
        Predict failure probability for a miner.

        Args:
            miner_id: Identifier of the miner
            features: Feature dictionary from miner profiler

        Returns:
            FailurePrediction with probability and risk assessment
        """
        if not self.is_trained:
            # Use heuristic prediction when model isn't trained
            return self._heuristic_prediction(miner_id, features)

        # Prepare features
        X = self._validate_features(features)
        X_scaled = self.scaler.transform(X)

        # Get prediction
        failure_probability = self.model.predict_proba(X_scaled)[0, 1]

        # Determine risk level
        risk_level = self._classify_risk(failure_probability)

        # Calculate confidence based on model certainty
        confidence = self._calculate_confidence(failure_probability)

        # Estimate time to failure
        time_to_failure = self._estimate_time_to_failure(failure_probability, features)

        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(features)

        # Generate recommendation
        recommendation = self._generate_recommendation(risk_level, contributing_factors)

        return FailurePrediction(
            miner_id=miner_id,
            timestamp=time.time(),
            failure_probability=float(failure_probability),
            risk_level=risk_level,
            confidence=confidence,
            time_to_failure_estimate=time_to_failure,
            contributing_factors=contributing_factors,
            recommendation=recommendation,
        )

    def _heuristic_prediction(
        self, miner_id: str, features: Dict[str, float]
    ) -> FailurePrediction:
        """Heuristic-based prediction when model isn't trained"""
        score = 0.0
        factors = []

        # Latency factors
        if features.get("latency_mean", 0) > 500:
            score += 0.2
            factors.append({"factor": "high_latency", "weight": 0.2})

        if features.get("latency_trend", 0) > 5:
            score += 0.15
            factors.append({"factor": "increasing_latency", "weight": 0.15})

        # Error factors
        if features.get("error_rate", 0) > 5:
            score += 0.25
            factors.append({"factor": "high_error_rate", "weight": 0.25})

        # Heartbeat factors
        if features.get("missed_heartbeats", 0) > 2:
            score += 0.2
            factors.append({"factor": "missed_heartbeats", "weight": 0.2})

        # Historical factors
        if features.get("historical_failures", 0) > 3:
            score += 0.15
            factors.append({"factor": "failure_history", "weight": 0.15})

        if features.get("reliability_score", 100) < 70:
            score += 0.1
            factors.append({"factor": "low_reliability", "weight": 0.1})

        failure_probability = min(1.0, score)
        risk_level = self._classify_risk(failure_probability)

        return FailurePrediction(
            miner_id=miner_id,
            timestamp=time.time(),
            failure_probability=failure_probability,
            risk_level=risk_level,
            confidence=0.5,  # Lower confidence for heuristic
            time_to_failure_estimate=None,
            contributing_factors=factors,
            recommendation=self._generate_recommendation(risk_level, factors),
        )

    def _classify_risk(self, probability: float) -> FailureRisk:
        """Classify risk level based on failure probability"""
        if probability >= 0.9:
            return FailureRisk.IMMINENT
        elif probability >= 0.7:
            return FailureRisk.HIGH
        elif probability >= 0.5:
            return FailureRisk.MODERATE
        elif probability >= 0.3:
            return FailureRisk.LOW
        else:
            return FailureRisk.MINIMAL

    def _calculate_confidence(self, probability: float) -> float:
        """Calculate prediction confidence"""
        # Higher confidence when probability is more extreme
        distance_from_midpoint = abs(probability - 0.5)
        confidence = 0.5 + distance_from_midpoint
        return min(0.95, confidence)

    def _estimate_time_to_failure(
        self, probability: float, features: Dict[str, float]
    ) -> Optional[float]:
        """Estimate time until failure (if predictable)"""
        if probability < 0.5:
            return None  # Failure not expected soon

        # Base estimate on probability and trend
        base_time = (1 - probability) * 3600  # Max 1 hour when probability is low

        # Adjust for latency trend
        trend = features.get("latency_trend", 0)
        if trend > 0:
            base_time *= max(0.1, 1 - trend / 20)

        # Adjust for recent failures
        recent_failures = features.get("historical_failures", 0)
        if recent_failures > 0:
            base_time *= max(0.2, 1 - recent_failures * 0.1)

        return max(60, base_time)  # Minimum 60 seconds

    def _identify_contributing_factors(
        self, features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify top contributing factors to failure risk"""
        factors = []

        # Check each feature against thresholds
        thresholds = {
            "latency_mean": (200, "High latency"),
            "latency_std": (100, "Latency variance"),
            "latency_trend": (5, "Increasing latency"),
            "error_rate": (2, "Error rate"),
            "missed_heartbeats": (1, "Missed heartbeats"),
            "reliability_score": (80, "Low reliability", True),  # Inverse threshold
        }

        for feature, config in thresholds.items():
            value = features.get(feature, 0)

            if len(config) == 3:  # Inverse threshold
                threshold, description, _ = config
                if value < threshold:
                    importance = self.feature_importance.get(feature, 0.1)
                    factors.append(
                        {
                            "factor": feature,
                            "description": description,
                            "value": value,
                            "threshold": threshold,
                            "importance": importance,
                        }
                    )
            else:
                threshold, description = config
                if value > threshold:
                    importance = self.feature_importance.get(feature, 0.1)
                    factors.append(
                        {
                            "factor": feature,
                            "description": description,
                            "value": value,
                            "threshold": threshold,
                            "importance": importance,
                        }
                    )

        # Sort by importance
        factors.sort(key=lambda x: x.get("importance", 0), reverse=True)

        return factors[:5]  # Top 5 factors

    def _generate_recommendation(
        self, risk_level: FailureRisk, factors: List[Dict[str, Any]]
    ) -> str:
        """Generate recommendation based on risk and factors"""
        if risk_level == FailureRisk.IMMINENT:
            return (
                "CRITICAL: Immediate traffic rerouting required. Node failure imminent."
            )

        if risk_level == FailureRisk.HIGH:
            return "HIGH RISK: Significantly reduce traffic to this node. Prepare backup routing."

        if risk_level == FailureRisk.MODERATE:
            factor_str = ", ".join(f.get("description", "") for f in factors[:2])
            return f"MODERATE RISK: Reduce routing weight. Key concerns: {factor_str}"

        if risk_level == FailureRisk.LOW:
            return "LOW RISK: Continue monitoring. Minor degradation signals detected."

        return "MINIMAL RISK: Node operating normally. No action required."

    def save_model(self, path: str) -> None:
        """Save trained model to disk"""
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.FEATURE_NAMES,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history,
        }

        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> bool:
        """Load model from disk"""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_importance = model_data.get("feature_importance", {})
            self.training_history = model_data.get("training_history", [])
            self.is_trained = True

            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()

    def export_prediction(self, prediction: FailurePrediction) -> Dict[str, Any]:
        """Export prediction as JSON-serializable dict"""
        return {
            "miner_id": prediction.miner_id,
            "timestamp": prediction.timestamp,
            "failure_probability": prediction.failure_probability,
            "risk_level": prediction.risk_level.value,
            "confidence": prediction.confidence,
            "time_to_failure_estimate": prediction.time_to_failure_estimate,
            "contributing_factors": prediction.contributing_factors,
            "recommendation": prediction.recommendation,
        }


# Singleton instance
_predictor_instance: Optional[FailurePredictor] = None


def get_failure_predictor() -> FailurePredictor:
    """Get or create the global failure predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FailurePredictor()
    return _predictor_instance