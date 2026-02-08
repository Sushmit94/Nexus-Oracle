import asyncio
import json
import time
import logging
import math
import random
from typing import Dict, Set, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ============ Configuration ============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pro-dashboard")

app = FastAPI(title="Predictive Router Oracle Dashboard")

# Strict CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 1. WebSocket Manager ============
class ConnectionManager:
    """Manages WebSocket connections and handles broadcasting."""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcasts message to all clients.
        Handles stale connections gracefully.
        """
        if not self.active_connections:
            return

        # Convert to JSON once to save CPU
        try:
            message_json = json.dumps(message)
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            return

        to_remove = set()
        for connection in self.active_connections:
            try:
                # We use send_text because we already serialized it
                await connection.send_text(message_json)
            except (WebSocketDisconnect, RuntimeError) as e:
                # RuntimeError often happens if connection is closed during send
                logger.warning(f"Failed to send to client: {e}")
                to_remove.add(connection)
            except Exception as e:
                logger.error(f"Unexpected broadcast error: {e}")
                to_remove.add(connection)

        for conn in to_remove:
            self.disconnect(conn)

manager = ConnectionManager()

# ============ 2. Data Simulator (Stable) ============
class DataSimulator:
    def __init__(self):
        self.tick = 0
        self.miners_config = {
            f"miner_{i:03d}": {"base_health": random.randint(50, 98)} 
            for i in range(1, 6)
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        self.tick += 1
        miners_data = {}
        
        # Generate Miner Data
        for mid, config in self.miners_config.items():
            # Create organic-looking oscillation
            oscillation = math.sin(self.tick * 0.1) * 10
            noise = random.uniform(-2, 2)
            current_health = max(0, min(100, config["base_health"] + oscillation + noise))
            
            # Determine status
            if current_health > 80: status = "healthy"
            elif current_health > 50: status = "degraded"
            else: status = "critical"

            miners_data[mid] = {
                "endpoint": f"http://node-{mid}.cortensor.io",
                "status": status,
                "health_score": int(current_health),
                "failure_probability": round((100 - current_health) / 100, 2),
                "routing_weight": round(current_health / 100, 2),
                # Latency correlates inversely with health
                "latency_ms": int(20 + (100 - current_health) * 2 + random.uniform(0, 10)),
                "last_heartbeat": time.time()
            }

        # Aggregates
        total = len(miners_data)
        healthy = sum(1 for m in miners_data.values() if m['status'] == 'healthy')
        degraded = sum(1 for m in miners_data.values() if m['status'] == 'degraded')
        critical = sum(1 for m in miners_data.values() if m['status'] == 'critical')
        
        # Safe average calculation
        avg_health = sum(m['health_score'] for m in miners_data.values()) / total if total > 0 else 0
        avg_latency = sum(m['latency_ms'] for m in miners_data.values()) / total if total > 0 else 0

        # Predictions Logic
        at_risk_miners = [
            {"miner_id": k, "probability": v["failure_probability"]} 
            for k, v in miners_data.items() 
            if v["failure_probability"] > 0.4
        ]
        at_risk_miners.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "timestamp": time.time(),
            "summary": {
                "total_miners": total,
                "healthy": healthy,
                "degraded": degraded,
                "critical": critical,
                "avg_health_score": int(avg_health),
                "avg_latency_ms": int(avg_latency)
            },
            "miners": miners_data,
            "predictions": {
                "miners_at_risk": at_risk_miners,
                "total_at_risk": len(at_risk_miners)
            },
            "recent_events": [
                {"type": "info", "message": "System operational", "time": time.time()}
            ]
        }

simulator = DataSimulator()

# ============ 3. API Routes ============
@app.get("/api/dashboard")
async def get_dashboard():
    return simulator.get_dashboard_data()

# ============ 4. WebSocket Handler (The Fix) ============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # 1. Send immediate data so the UI populates instantly
        await websocket.send_json(simulator.get_dashboard_data())
        
        # 2. Keep the connection open and listen for client pings
        # We DO NOT send data from this loop. We only listen.
        while True:
            data = await websocket.receive_text()
            # Optional: Handle client-side heartbeat
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket Connection Error: {e}")
        manager.disconnect(websocket)

# ============ 5. Background Broadcaster ============
async def broadcast_loop():
    """
    This is the ONLY place that pushes updates to clients.
    Running every 2 seconds.
    """
    logger.info("Starting broadcast loop...")
    while True:
        try:
            await asyncio.sleep(2)
            data = simulator.get_dashboard_data()
            await manager.broadcast({"type": "update", "data": data})
        except Exception as e:
            logger.error(f"Broadcast loop error: {e}")
            await asyncio.sleep(5) # Backoff on error

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)