"""
Failure Simulation Script
Simulates various failure scenarios for testing the prediction system
"""

import asyncio
import random
import time
import logging
import argparse
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures to simulate"""

    LATENCY_SPIKE = "latency_spike"
    GRADUAL_DEGRADATION = "gradual_degradation"
    RANDOM_ERRORS = "random_errors"
    HEARTBEAT_LOSS = "heartbeat_loss"
    CASCADING_FAILURE = "cascading_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"


@dataclass
class SimulatedMiner:
    """Simulated miner with configurable behavior"""

    miner_id: str
    base_latency: float = 50.0
    base_error_rate: float = 0.01
    base_throughput: float = 100.0

    # Current state (affected by failures)
    current_latency: float = 50.0
    current_error_rate: float = 0.01
    current_throughput: float = 100.0
    is_alive: bool = True

    # Failure state
    active_failures: List[str] = None

    def __post_init__(self):
        if self.active_failures is None:
            self.active_failures = []

    def reset(self):
        """Reset to baseline state"""
        self.current_latency = self.base_latency
        self.current_error_rate = self.base_error_rate
        self.current_throughput = self.base_throughput
        self.is_alive = True
        self.active_failures = []


class FailureSimulator:
    """
    Simulates various failure scenarios for testing the prediction system.
    """

    def __init__(self, num_miners: int = 5):
        self.miners: Dict[str, SimulatedMiner] = {}
        self._init_miners(num_miners)
        self.running = False
        self.tick = 0

    def _init_miners(self, num_miners: int):
        """Initialize simulated miners"""
        for i in range(num_miners):
            miner_id = f"miner_{i + 1:03d}"
            self.miners[miner_id] = SimulatedMiner(
                miner_id=miner_id,
                base_latency=30 + random.uniform(0, 40),
                base_error_rate=random.uniform(0.001, 0.02),
                base_throughput=80 + random.uniform(0, 40),
            )
        logger.info(f"Initialized {num_miners} simulated miners")

    def inject_failure(
        self,
        miner_id: str,
        failure_type: FailureType,
        severity: float = 1.0,
        duration: float = None,
    ) -> bool:
        """
        Inject a failure into a miner.

        Args:
            miner_id: Target miner
            failure_type: Type of failure to inject
            severity: 0-1, intensity of failure
            duration: Duration in seconds (None for permanent)
        """
        if miner_id not in self.miners:
            logger.error(f"Unknown miner: {miner_id}")
            return False

        miner = self.miners[miner_id]

        logger.warning(
            f"Injecting {failure_type.value} into {miner_id} (severity={severity})"
        )

        if failure_type == FailureType.LATENCY_SPIKE:
            miner.current_latency = miner.base_latency * (1 + severity * 10)

        elif failure_type == FailureType.GRADUAL_DEGRADATION:
            # Start gradual degradation (handled in tick)
            miner.active_failures.append(f"degradation_{severity}")

        elif failure_type == FailureType.RANDOM_ERRORS:
            miner.current_error_rate = min(1.0, miner.base_error_rate + severity * 0.2)

        elif failure_type == FailureType.HEARTBEAT_LOSS:
            miner.is_alive = False

        elif failure_type == FailureType.RESOURCE_EXHAUSTION:
            miner.current_throughput = miner.base_throughput * (1 - severity * 0.8)
            miner.current_latency = miner.base_latency * (1 + severity * 5)

        elif failure_type == FailureType.NETWORK_PARTITION:
            miner.is_alive = random.random() > severity  # Intermittent
            miner.current_latency = (
                miner.base_latency * (1 + severity * 20)
                if miner.is_alive
                else float("inf")
            )

        miner.active_failures.append(failure_type.value)

        # Schedule recovery if duration specified
        if duration:
            asyncio.create_task(
                self._schedule_recovery(miner_id, failure_type, duration)
            )

        return True

    async def _schedule_recovery(
        self, miner_id: str, failure_type: FailureType, duration: float
    ):
        """Schedule automatic recovery after duration"""
        await asyncio.sleep(duration)
        self.recover_miner(miner_id, failure_type)

    def recover_miner(self, miner_id: str, failure_type: FailureType = None):
        """Recover a miner from failure"""
        if miner_id not in self.miners:
            return

        miner = self.miners[miner_id]

        if failure_type:
            # Recover from specific failure
            if failure_type.value in miner.active_failures:
                miner.active_failures.remove(failure_type.value)
                logger.info(f"Recovered {miner_id} from {failure_type.value}")

        if not miner.active_failures:
            # Full recovery
            miner.reset()
            logger.info(f"Full recovery for {miner_id}")

    def inject_cascading_failure(self, initial_miner: str, probability: float = 0.5):
        """Simulate cascading failure across miners"""
        if initial_miner not in self.miners:
            return

        logger.critical(f"Initiating cascading failure from {initial_miner}")

        # Affect initial miner severely
        self.inject_failure(initial_miner, FailureType.LATENCY_SPIKE, severity=1.0)

        # Cascade to neighbors
        for miner_id in self.miners:
            if miner_id != initial_miner and random.random() < probability:
                severity = random.uniform(0.3, 0.8)
                self.inject_failure(
                    miner_id, FailureType.RESOURCE_EXHAUSTION, severity=severity
                )

    def tick_simulation(self) -> Dict[str, Any]:
        """
        Advance simulation by one tick and return current state.
        Returns metrics for all miners.
        """
        self.tick += 1
        metrics = {}

        for miner_id, miner in self.miners.items():
            # Apply gradual degradation if active
            for failure in miner.active_failures:
                if failure.startswith("degradation_"):
                    severity = float(failure.split("_")[1])
                    miner.current_latency *= 1 + severity * 0.01
                    miner.current_throughput *= 1 - severity * 0.005

            # Add noise
            latency = miner.current_latency * random.uniform(0.9, 1.1)
            error_rate = miner.current_error_rate * random.uniform(0.8, 1.2)
            throughput = miner.current_throughput * random.uniform(0.95, 1.05)

            metrics[miner_id] = {
                "latency_ms": round(latency, 2),
                "error_rate": round(min(1.0, error_rate), 4),
                "throughput_rps": round(max(0, throughput), 2),
                "is_alive": miner.is_alive,
                "active_failures": miner.active_failures.copy(),
                "health_score": self._calculate_health(
                    latency, error_rate, throughput, miner.is_alive
                ),
            }

        return metrics

    def _calculate_health(
        self, latency: float, error_rate: float, throughput: float, is_alive: bool
    ) -> float:
        """Calculate health score from metrics"""
        if not is_alive:
            return 0.0

        latency_score = max(0, 100 - latency / 10)
        error_score = max(0, 100 - error_rate * 1000)
        throughput_score = min(100, throughput)

        return round((latency_score + error_score + throughput_score) / 3, 1)

    def get_scenario_metrics(self, scenario: str) -> Dict[str, Any]:
        """Get metrics for a predefined scenario"""
        scenarios = {
            "healthy": lambda: None,  # No failures
            "single_degraded": lambda: self.inject_failure(
                random.choice(list(self.miners.keys())),
                FailureType.GRADUAL_DEGRADATION,
                severity=0.5,
            ),
            "multi_failure": lambda: [
                self.inject_failure(mid, FailureType.RANDOM_ERRORS, severity=0.3)
                for mid in random.sample(
                    list(self.miners.keys()), min(3, len(self.miners))
                )
            ],
            "cascade": lambda: self.inject_cascading_failure(
                random.choice(list(self.miners.keys()))
            ),
            "network_issues": lambda: [
                self.inject_failure(mid, FailureType.NETWORK_PARTITION, severity=0.5)
                for mid in random.sample(list(self.miners.keys()), 2)
            ],
        }

        # Reset all miners first
        for miner in self.miners.values():
            miner.reset()

        # Apply scenario
        if scenario in scenarios:
            scenarios[scenario]()

        return self.tick_simulation()

    async def run_simulation_loop(self, interval: float = 1.0, callback=None):
        """Run continuous simulation"""
        self.running = True
        logger.info("Starting simulation loop")

        while self.running:
            metrics = self.tick_simulation()

            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)

            await asyncio.sleep(interval)

    def stop(self):
        """Stop simulation"""
        self.running = False
        logger.info("Simulation stopped")


# ============ CLI Interface ============


def print_metrics(metrics: Dict[str, Any]):
    """Pretty print metrics"""
    print("\n" + "=" * 60)
    print(f"Simulation Tick - {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    for miner_id, data in metrics.items():
        status = (
            "🟢"
            if data["is_alive"] and data["health_score"] > 70
            else ("🟡" if data["health_score"] > 40 else "🔴")
        )

        failures = (
            ", ".join(data["active_failures"]) if data["active_failures"] else "none"
        )

        print(
            f"{status} {miner_id}: "
            f"Health={data['health_score']:.0f}% "
            f"Latency={data['latency_ms']:.0f}ms "
            f"Errors={data['error_rate'] * 100:.2f}% "
            f"Failures=[{failures}]"
        )


async def run_interactive_simulation():
    """Run interactive simulation"""
    simulator = FailureSimulator(num_miners=5)

    print("\n🎮 Failure Simulation Console")
    print("=" * 40)
    print("Commands:")
    print("  1. Inject latency spike")
    print("  2. Inject gradual degradation")
    print("  3. Inject random errors")
    print("  4. Kill miner (heartbeat loss)")
    print("  5. Inject cascading failure")
    print("  6. Recover all miners")
    print("  7. Show current state")
    print("  8. Run continuous simulation")
    print("  q. Quit")
    print("=" * 40)

    while True:
        try:
            cmd = input("\n> Enter command: ").strip().lower()

            if cmd == "q":
                break
            elif cmd == "1":
                miner = input("  Miner ID (e.g., miner_001): ")
                simulator.inject_failure(
                    miner, FailureType.LATENCY_SPIKE, severity=1.0, duration=30
                )
            elif cmd == "2":
                miner = input("  Miner ID: ")
                simulator.inject_failure(
                    miner, FailureType.GRADUAL_DEGRADATION, severity=0.5
                )
            elif cmd == "3":
                miner = input("  Miner ID: ")
                simulator.inject_failure(
                    miner, FailureType.RANDOM_ERRORS, severity=0.5, duration=20
                )
            elif cmd == "4":
                miner = input("  Miner ID: ")
                simulator.inject_failure(miner, FailureType.HEARTBEAT_LOSS)
            elif cmd == "5":
                miner = input("  Starting miner ID: ")
                simulator.inject_cascading_failure(miner)
            elif cmd == "6":
                for miner_id in simulator.miners:
                    simulator.miners[miner_id].reset()
                print("  All miners recovered")
            elif cmd == "7":
                print_metrics(simulator.tick_simulation())
            elif cmd == "8":
                print("  Running continuous simulation (Ctrl+C to stop)...")
                try:
                    while True:
                        print_metrics(simulator.tick_simulation())
                        await asyncio.sleep(2)
                except KeyboardInterrupt:
                    print("\n  Stopped")
            else:
                print("  Unknown command")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  Error: {e}")

    print("\n👋 Simulation ended")


def main():
    parser = argparse.ArgumentParser(
        description="Failure Simulation for Predictive Router Oracle"
    )
    parser.add_argument(
        "--miners", type=int, default=5, help="Number of simulated miners"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=[
            "healthy",
            "single_degraded",
            "multi_failure",
            "cascade",
            "network_issues",
        ],
        help="Run predefined scenario",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive mode"
    )
    parser.add_argument(
        "--continuous", action="store_true", help="Run continuous simulation"
    )

    args = parser.parse_args()

    if args.interactive:
        asyncio.run(run_interactive_simulation())
    elif args.scenario:
        simulator = FailureSimulator(num_miners=args.miners)
        metrics = simulator.get_scenario_metrics(args.scenario)
        print_metrics(metrics)
    elif args.continuous:
        simulator = FailureSimulator(num_miners=args.miners)

        async def continuous():
            while True:
                print_metrics(simulator.tick_simulation())
                await asyncio.sleep(2)

        try:
            asyncio.run(continuous())
        except KeyboardInterrupt:
            print("\nSimulation stopped")
    else:
        # Default: single tick
        simulator = FailureSimulator(num_miners=args.miners)
        print_metrics(simulator.tick_simulation())


if __name__ == "__main__":
    main()