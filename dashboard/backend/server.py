"""
Dashboard Backend Server
WebSocket-enabled server for real-time dashboard updates
"""

import asyncio
import json
import time
import logging
from typing import Dict, Set, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predictive Router Oracle Dashboard")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ WebSocket Manager ============


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message)
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception:
                disconnected.add(connection)

        for conn in disconnected:
            self.active_connections.discard(conn)


manager = ConnectionManager()


# ============ Sample Data Generator ============


class DataSimulator:
    """Simulates real-time data for demo purposes"""

    def __init__(self):
        self.miners = {
            "miner_001": {
                "endpoint": "http://node1.cortensor.io:8001",
                "base_health": 95,
            },
            "miner_002": {
                "endpoint": "http://node2.cortensor.io:8002",
                "base_health": 72,
            },
            "miner_003": {
                "endpoint": "http://node3.cortensor.io:8003",
                "base_health": 88,
            },
            "miner_004": {
                "endpoint": "http://node4.cortensor.io:8004",
                "base_health": 45,
            },
            "miner_005": {
                "endpoint": "http://node5.cortensor.io:8005",
                "base_health": 98,
            },
        }
        self.tick = 0

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Generate current dashboard state"""
        import random
        import math

        self.tick += 1

        miners_data = {}
        for miner_id, config in self.miners.items():
            # Add some variation
            noise = random.uniform(-5, 5)
            wave = math.sin(self.tick / 10) * 3

            health = max(0, min(100, config["base_health"] + noise + wave))
            failure_prob = (100 - health) / 100

            miners_data[miner_id] = {
                "endpoint": config["endpoint"],
                "health_score": round(health, 1),
                "failure_probability": round(failure_prob, 3),
                "routing_weight": round(1 - failure_prob, 3),
                "status": "healthy"
                if health > 70
                else ("degraded" if health > 40 else "critical"),
                "latency_ms": round(
                    50 + (100 - health) * 5 + random.uniform(-10, 10), 1
                ),
                "throughput_rps": round(
                    100 * (health / 100) + random.uniform(-5, 5), 1
                ),
                "error_rate": round((100 - health) / 10 + random.uniform(-0.5, 0.5), 2),
                "last_heartbeat": time.time() - random.uniform(0, 5),
            }

        # System summary
        total_miners = len(miners_data)
        healthy_count = sum(1 for m in miners_data.values() if m["status"] == "healthy")
        degraded_count = sum(
            1 for m in miners_data.values() if m["status"] == "degraded"
        )
        critical_count = sum(
            1 for m in miners_data.values() if m["status"] == "critical"
        )

        avg_health = sum(m["health_score"] for m in miners_data.values()) / total_miners
        avg_latency = sum(m["latency_ms"] for m in miners_data.values()) / total_miners

        return {
            "timestamp": time.time(),
            "summary": {
                "total_miners": total_miners,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "critical": critical_count,
                "avg_health_score": round(avg_health, 1),
                "avg_latency_ms": round(avg_latency, 1),
                "system_status": "healthy" if critical_count == 0 else "degraded",
            },
            "miners": miners_data,
            "recent_events": self._generate_events(),
            "predictions": self._generate_predictions(miners_data),
        }

    def _generate_events(self) -> list:
        """Generate sample recent events"""
        events = [
            {
                "time": time.time() - 60,
                "type": "info",
                "message": "Routing weights updated",
            },
            {
                "time": time.time() - 180,
                "type": "warning",
                "message": "miner_004 latency spike detected",
            },
            {
                "time": time.time() - 300,
                "type": "success",
                "message": "miner_003 recovered from degraded state",
            },
        ]
        return events[:5]

    def _generate_predictions(self, miners_data: Dict) -> Dict:
        """Generate prediction summary"""
        at_risk = [
            {"miner_id": mid, "probability": m["failure_probability"]}
            for mid, m in miners_data.items()
            if m["failure_probability"] > 0.3
        ]

        return {
            "miners_at_risk": sorted(at_risk, key=lambda x: -x["probability"]),
            "total_at_risk": len(at_risk),
            "highest_risk": at_risk[0] if at_risk else None,
        }


simulator = DataSimulator()


# ============ REST Endpoints ============


@app.get("/")
async def root():
    return {"name": "Predictive Router Oracle Dashboard", "status": "operational"}


@app.get("/api/dashboard")
async def get_dashboard():
    """Get current dashboard state"""
    return simulator.get_dashboard_data()


@app.get("/api/miners")
async def get_miners():
    """Get all miners"""
    data = simulator.get_dashboard_data()
    return data["miners"]


@app.get("/api/miners/{miner_id}")
async def get_miner(miner_id: str):
    """Get specific miner"""
    data = simulator.get_dashboard_data()
    if miner_id in data["miners"]:
        return data["miners"][miner_id]
    return {"error": "Miner not found"}


@app.get("/api/predictions")
async def get_predictions():
    """Get current predictions"""
    data = simulator.get_dashboard_data()
    return data["predictions"]


# ============ WebSocket Endpoint ============


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        # Send initial state
        await websocket.send_json(simulator.get_dashboard_data())

        # Listen for messages and send updates
        while True:
            try:
                # Wait for client message or timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)

                # Handle client commands
                try:
                    command = json.loads(data)
                    if command.get("type") == "ping":
                        await websocket.send_json(
                            {"type": "pong", "timestamp": time.time()}
                        )
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send periodic updates
                await websocket.send_json(
                    {"type": "update", "data": simulator.get_dashboard_data()}
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============ Background Update Task ============


async def broadcast_updates():
    """Background task to broadcast updates"""
    while True:
        await asyncio.sleep(3)
        await manager.broadcast(
            {"type": "update", "data": simulator.get_dashboard_data()}
        )


@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(broadcast_updates())
    logger.info("Dashboard backend started")


# ============ Run Server ============


def run_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Run the dashboard backend server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()