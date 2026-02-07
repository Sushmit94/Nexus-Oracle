"""
Demo Runner Script
Runs a complete demonstration of the Predictive Router Oracle system
"""

import asyncio
import time
import logging
import argparse
import signal
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DemoRunner:
    """
    Orchestrates a complete demonstration of the Predictive Router Oracle.
    """

    def __init__(self, mode: str = "full"):
        self.mode = mode
        self.running = False
        self.components = {}

    async def start_api_server(self, port: int = 8000):
        """Start the prediction API server"""
        logger.info(f"Starting API server on port {port}...")

        try:
            import uvicorn
            from api.prediction_api import app

            config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
            server = uvicorn.Server(config)
            self.components["api"] = server
            await server.serve()
        except ImportError:
            logger.warning("Could not import API modules. Running in demo mode.")

    async def start_dashboard(self, port: int = 8080):
        """Start the dashboard backend"""
        logger.info(f"Starting dashboard on port {port}...")

        try:
            import uvicorn
            from dashboard.backend.server import app

            config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
            server = uvicorn.Server(config)
            self.components["dashboard"] = server
            await server.serve()
        except ImportError:
            logger.warning("Could not import dashboard modules. Running in demo mode.")

    async def start_metrics_collection(self):
        """Start metrics collection from simulated miners"""
        logger.info("Starting metrics collection...")

        try:
            from scripts.simulate_failures import FailureSimulator

            simulator = FailureSimulator(num_miners=5)
            self.components["simulator"] = simulator

            async def collect_loop():
                while self.running:
                    metrics = simulator.tick_simulation()
                    logger.debug(f"Collected metrics from {len(metrics)} miners")
                    await asyncio.sleep(5)

            asyncio.create_task(collect_loop())
        except ImportError:
            logger.warning("Could not import simulator. Skipping metrics collection.")

    async def run_prediction_cycle(self):
        """Run continuous prediction cycles"""
        logger.info("Starting prediction cycle...")

        cycle_count = 0

        while self.running:
            cycle_count += 1
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Prediction Cycle #{cycle_count}")
            logger.info(f"{'=' * 50}")

            # Simulate prediction results
            predictions = self._generate_demo_predictions()

            for miner_id, pred in predictions.items():
                status_icon = (
                    "🔴"
                    if pred["failure_probability"] > 0.7
                    else ("🟡" if pred["failure_probability"] > 0.3 else "🟢")
                )

                logger.info(
                    f"{status_icon} {miner_id}: "
                    f"P(fail)={pred['failure_probability']:.1%}, "
                    f"Confidence={pred['confidence']:.1%}, "
                    f"Action={pred['action']}"
                )

            # Check for routing updates
            at_risk = [
                m for m, p in predictions.items() if p["failure_probability"] > 0.5
            ]
            if at_risk:
                logger.warning(
                    f"⚠️  {len(at_risk)} miner(s) at elevated risk: {', '.join(at_risk)}"
                )

            await asyncio.sleep(10)

    def _generate_demo_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Generate demo prediction data"""
        import random
        import math

        tick = time.time() / 30

        miners = {
            "miner_001": {"base_risk": 0.05},
            "miner_002": {"base_risk": 0.35},
            "miner_003": {"base_risk": 0.15},
            "miner_004": {"base_risk": 0.65},
            "miner_005": {"base_risk": 0.03},
        }

        predictions = {}
        for miner_id, config in miners.items():
            # Add variation
            noise = random.uniform(-0.1, 0.1)
            wave = math.sin(tick + hash(miner_id) % 10) * 0.1

            prob = max(0, min(1, config["base_risk"] + noise + wave))

            predictions[miner_id] = {
                "failure_probability": prob,
                "confidence": 0.7 + random.uniform(0, 0.25),
                "action": "REROUTE"
                if prob > 0.7
                else ("REDUCE" if prob > 0.4 else "MONITOR"),
            }

        return predictions

    async def run_demo_scenario(self, scenario: str = "standard"):
        """Run a specific demo scenario"""
        logger.info(f"\n🎬 Starting demo scenario: {scenario}")

        scenarios = {
            "standard": self._scenario_standard,
            "failure": self._scenario_failure,
            "recovery": self._scenario_recovery,
            "cascade": self._scenario_cascade,
        }

        if scenario in scenarios:
            await scenarios[scenario]()
        else:
            logger.error(f"Unknown scenario: {scenario}")

    async def _scenario_standard(self):
        """Standard demo - show normal operation"""
        logger.info("📊 Demonstrating normal operation...")

        for i in range(5):
            logger.info(f"\n--- Tick {i + 1}/5 ---")
            predictions = self._generate_demo_predictions()

            healthy = sum(
                1 for p in predictions.values() if p["failure_probability"] < 0.3
            )
            at_risk = sum(
                1 for p in predictions.values() if 0.3 <= p["failure_probability"] < 0.7
            )
            critical = sum(
                1 for p in predictions.values() if p["failure_probability"] >= 0.7
            )

            logger.info(
                f"System Status: 🟢 Healthy={healthy} 🟡 At Risk={at_risk} 🔴 Critical={critical}"
            )

            await asyncio.sleep(2)

        logger.info("\n✅ Standard scenario complete")

    async def _scenario_failure(self):
        """Demo showing failure detection and response"""
        logger.info("🔥 Demonstrating failure detection...")

        # Simulate gradual degradation
        for i in range(8):
            degradation = i / 7

            logger.info(f"\n--- Degradation Level: {degradation:.0%} ---")

            if degradation < 0.3:
                logger.info("📊 Status: Normal operation")
            elif degradation < 0.5:
                logger.warning("⚠️  Warning: Elevated latency detected on miner_004")
            elif degradation < 0.7:
                logger.warning("⚠️  Alert: miner_004 showing signs of degradation")
                logger.info("🔄 Routing: Reducing weight for miner_004 (1.0 → 0.6)")
            else:
                logger.error(
                    "🔴 Critical: miner_004 failure predicted with 85% confidence"
                )
                logger.info(
                    "🔄 Routing: Emergency reroute - miner_004 weight set to 0.1"
                )
                logger.info(
                    "📡 Oracle: Publishing emergency update to on-chain registry"
                )

            await asyncio.sleep(1.5)

        logger.info("\n✅ Failure scenario complete - traffic successfully rerouted")

    async def _scenario_recovery(self):
        """Demo showing recovery after failure"""
        logger.info("🔧 Demonstrating recovery process...")

        # Show recovery timeline
        stages = [
            ("🔴 Initial State", "miner_004 offline, weight=0.0"),
            ("🟡 Detection", "Heartbeat detected from miner_004"),
            ("🟡 Verification", "Running health checks..."),
            ("🟡 Gradual Recovery", "Weight increased to 0.3"),
            ("🟢 Monitoring", "Stable for 2 minutes, weight=0.6"),
            ("🟢 Full Recovery", "Weight restored to 1.0"),
        ]

        for status, message in stages:
            logger.info(f"\n{status}: {message}")
            await asyncio.sleep(2)

        logger.info("\n✅ Recovery scenario complete")

    async def _scenario_cascade(self):
        """Demo showing cascading failure prevention"""
        logger.info("🌊 Demonstrating cascading failure prevention...")

        logger.info("\n--- Initial Failure ---")
        logger.error("🔴 miner_002 experiencing critical latency spike")
        await asyncio.sleep(1)

        logger.info("\n--- Cascade Detection ---")
        logger.warning("⚠️  AI detects potential cascade risk")
        logger.info("📊 Analyzing: miner_002 handles 30% of traffic")
        logger.info(
            "📊 Risk Assessment: High probability of overloading miner_003, miner_005"
        )
        await asyncio.sleep(1.5)

        logger.info("\n--- Proactive Mitigation ---")
        logger.info("🔄 Preemptively reducing weight on connected miners")
        logger.info("   miner_003: 1.0 → 0.7 (preventive)")
        logger.info("   miner_005: 1.0 → 0.8 (preventive)")
        await asyncio.sleep(1.5)

        logger.info("\n--- Gradual Redistribution ---")
        logger.info("🔄 Redistributing traffic to healthy nodes")
        logger.info("   miner_001: weight increased to 1.2")
        logger.info("   miner_004: weight increased to 1.1")
        await asyncio.sleep(1.5)

        logger.info("\n--- Cascade Prevented ---")
        logger.info("✅ System stabilized without cascade")
        logger.info("📊 Total affected requests: 0 (proactive mitigation)")

        logger.info("\n✅ Cascade prevention scenario complete")

    async def run_full_demo(self):
        """Run the complete demo"""
        self.running = True

        print("\n" + "=" * 60)
        print("🚀 PREDICTIVE ROUTER ORACLE - DEMONSTRATION")
        print("=" * 60)
        print("\nThis demo showcases the AI-driven routing oracle system")
        print("that predicts and prevents miner/validator failures.\n")

        # Run scenarios
        scenarios = ["standard", "failure", "recovery", "cascade"]

        for scenario in scenarios:
            await self.run_demo_scenario(scenario)

            if scenario != scenarios[-1]:
                print("\n" + "-" * 40)
                print("Press Enter for next scenario...")
                await asyncio.sleep(3)

        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETE")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("  ✅ Real-time miner health monitoring")
        print("  ✅ Multi-model AI prediction (Statistical + ML + LLM)")
        print("  ✅ Agent Arbitration Nexus for consensus")
        print("  ✅ Dynamic routing weight adjustment")
        print("  ✅ Cascading failure prevention")
        print("  ✅ Automatic recovery handling")
        print("\nFor production deployment, see the README.md")
        print("=" * 60 + "\n")

    async def start_all(self):
        """Start all components"""
        self.running = True

        tasks = []

        if self.mode in ["full", "api"]:
            tasks.append(asyncio.create_task(self.start_api_server()))

        if self.mode in ["full", "dashboard"]:
            tasks.append(asyncio.create_task(self.start_dashboard()))

        if self.mode in ["full", "prediction"]:
            await self.start_metrics_collection()
            tasks.append(asyncio.create_task(self.run_prediction_cycle()))

        if self.mode == "demo":
            await self.run_full_demo()
            return

        if tasks:
            await asyncio.gather(*tasks)

    def stop(self):
        """Stop all components"""
        self.running = False
        logger.info("Stopping all components...")


def main():
    parser = argparse.ArgumentParser(description="Predictive Router Oracle Demo Runner")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "api", "dashboard", "prediction", "demo"],
        default="demo",
        help="Components to run",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["standard", "failure", "recovery", "cascade"],
        help="Run specific scenario",
    )

    args = parser.parse_args()

    runner = DemoRunner(mode=args.mode)

    # Handle shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run
    if args.scenario:
        asyncio.run(runner.run_demo_scenario(args.scenario))
    else:
        asyncio.run(runner.start_all())


if __name__ == "__main__":
    main()