"""
Prediction API Module
FastAPI-based paid API for accessing predictions and routing data
"""

# ============ 1. ALL IMPORTS AT THE TOP ============
import time
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Header, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .auth import AuthManager, get_auth_manager, ApiKey, Permission, AccessTier

# ============ 2. LOGGING CONFIGURATION ============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ 3. CREATE FASTAPI APP FIRST ============
app = FastAPI(
    title="Predictive Router Oracle API",
    description="AI-driven routing oracle that predicts miner/validator failures",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 4. PYDANTIC MODELS ============

class HealthResponse(BaseModel):
    miner_id: str
    health_score: float
    failure_probability: float
    routing_weight: float
    is_healthy: bool
    last_updated: float
    latency_mean: Optional[float] = None
    error_rate: Optional[float] = None


class PredictionResponse(BaseModel):
    miner_id: str
    timestamp: float
    failure_probability: float
    confidence: float
    risk_level: str
    recommended_action: str
    reasoning: str
    agent_predictions: Optional[List[Dict[str, Any]]] = None
    time_to_failure_estimate: Optional[float] = None


class RoutingTableResponse(BaseModel):
    timestamp: float
    total_miners: int
    healthy_miners: int
    routing_strategy: str
    miners: Dict[str, Dict[str, Any]]


class DecisionResponse(BaseModel):
    miner_id: str
    timestamp: float
    old_weight: float
    new_weight: float
    reason: str
    applied: bool


class ApiKeyRequest(BaseModel):
    name: str
    tier: str = "basic"
    expires_in_days: Optional[int] = None
    owner_email: Optional[str] = None


class ApiKeyResponse(BaseModel):
    key_id: str
    api_key: str
    tier: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    retry_after: Optional[float] = None


# ============ 5. CONNECTION MANAGER ============

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# ============ 6. SIMULATED DATA STORE ============

class DataStore:
    """Simulated data store for API responses"""

    def __init__(self):
        self.miners = {}
        self.predictions = {}
        self.decisions = []
        self._init_demo_data()

    def _init_demo_data(self):
        """Initialize demo data"""
        demo_miners = [
            ("miner_001", "http://node1.cortensor.io:8001", 0.95, 0.08),
            ("miner_002", "http://node2.cortensor.io:8002", 0.72, 0.35),
            ("miner_003", "http://node3.cortensor.io:8003", 0.88, 0.15),
            ("miner_004", "http://node4.cortensor.io:8004", 0.45, 0.72),
            ("miner_005", "http://node5.cortensor.io:8005", 0.98, 0.03),
        ]

        for miner_id, endpoint, health, fail_prob in demo_miners:
            self.miners[miner_id] = {
                "endpoint": endpoint,
                "health_score": health * 100,
                "failure_probability": fail_prob,
                "routing_weight": 1 - fail_prob,
                "is_healthy": fail_prob < 0.5,
                "latency_mean": 50 + fail_prob * 400,
                "error_rate": fail_prob * 8,
                "last_updated": time.time(),
            }

            self.predictions[miner_id] = {
                "failure_probability": fail_prob,
                "confidence": 0.85,
                "risk_level": "high" if fail_prob > 0.5 else "low",
                "recommended_action": "reroute" if fail_prob > 0.7 else "monitor",
                "reasoning": f"Failure probability is {fail_prob * 100:.0f}%",
                "timestamp": time.time(),
            }


data_store = DataStore()


# ============ 7. DEPENDENCIES ============

async def verify_api_key(
    x_api_key: str = Header(..., description="API Key for authentication"),
) -> ApiKey:
    """Verify API key and check rate limits"""
    auth = get_auth_manager()

    # Validate key
    api_key = auth.validate_key(x_api_key)
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")

    # Check rate limit
    allowed, rate_info = auth.check_rate_limit(api_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({rate_info.get('exceeded')})",
            headers={"Retry-After": str(int(rate_info.get("retry_after", 60)))},
        )

    return api_key


def require_permission(permission: Permission):
    """Dependency factory for permission checking"""

    async def check_permission(api_key: ApiKey = Depends(verify_api_key)) -> ApiKey:
        auth = get_auth_manager()
        if not auth.check_permission(api_key, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission.value} required",
            )
        return api_key

    return check_permission


# ============ 8. HEALTH ENDPOINTS ============

@app.get("/", tags=["Health"])
async def root():
    """API root endpoint"""
    return {
        "name": "Predictive Router Oracle API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "miners_tracked": len(data_store.miners),
    }


# ============ 9. WEBSOCKET ENDPOINT (NOW app IS DEFINED) ============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and send updates every 2 seconds
            # In a real app, you'd trigger this based on actual data updates
            await websocket.receive_text()  # Wait for messages (optional)
            
            # Example: sending the current routing table as an update
            # You can customize this payload to match your DashboardData type
            await manager.broadcast({
                "type": "update",
                "data": {
                    "summary": {
                        "active_miners": len(data_store.miners),
                        "total_requests": 15420,  # Example data
                        "system_health": 98.5
                    }
                }
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============ 10. MINER HEALTH ENDPOINTS ============

@app.get("/api/v1/miners", response_model=Dict[str, HealthResponse], tags=["Miners"])
async def get_all_miners(
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(require_permission(Permission.READ_HEALTH)),
):
    """Get health status for all registered miners"""
    auth = get_auth_manager()
    auth.record_request(api_key, "/api/v1/miners")

    result = {}
    for miner_id, data in data_store.miners.items():
        result[miner_id] = HealthResponse(
            miner_id=miner_id,
            health_score=data["health_score"],
            failure_probability=data["failure_probability"],
            routing_weight=data["routing_weight"],
            is_healthy=data["is_healthy"],
            last_updated=data["last_updated"],
            latency_mean=data["latency_mean"],
            error_rate=data["error_rate"],
        )

    return result


@app.get("/api/v1/miners/{miner_id}", response_model=HealthResponse, tags=["Miners"])
async def get_miner_health(
    miner_id: str, api_key: ApiKey = Depends(require_permission(Permission.READ_HEALTH))
):
    """Get health status for a specific miner"""
    auth = get_auth_manager()
    auth.record_request(api_key, f"/api/v1/miners/{miner_id}")

    if miner_id not in data_store.miners:
        raise HTTPException(status_code=404, detail=f"Miner not found: {miner_id}")

    data = data_store.miners[miner_id]
    return HealthResponse(
        miner_id=miner_id,
        health_score=data["health_score"],
        failure_probability=data["failure_probability"],
        routing_weight=data["routing_weight"],
        is_healthy=data["is_healthy"],
        last_updated=data["last_updated"],
        latency_mean=data["latency_mean"],
        error_rate=data["error_rate"],
    )


# ============ 11. PREDICTION ENDPOINTS ============

@app.get(
    "/api/v1/predictions/{miner_id}",
    response_model=PredictionResponse,
    tags=["Predictions"],
)
async def get_miner_prediction(
    miner_id: str,
    include_agents: bool = Query(
        False, description="Include individual agent predictions"
    ),
    api_key: ApiKey = Depends(require_permission(Permission.READ_PREDICTIONS)),
):
    """Get failure prediction for a specific miner"""
    auth = get_auth_manager()
    auth.record_request(api_key, f"/api/v1/predictions/{miner_id}")

    if miner_id not in data_store.predictions:
        raise HTTPException(status_code=404, detail=f"Prediction not found: {miner_id}")

    pred = data_store.predictions[miner_id]

    response = PredictionResponse(
        miner_id=miner_id,
        timestamp=pred["timestamp"],
        failure_probability=pred["failure_probability"],
        confidence=pred["confidence"],
        risk_level=pred["risk_level"],
        recommended_action=pred["recommended_action"],
        reasoning=pred["reasoning"],
    )

    if include_agents:
        response.agent_predictions = [
            {"agent": "statistical", "probability": pred["failure_probability"] * 0.9},
            {"agent": "ml", "probability": pred["failure_probability"]},
            {"agent": "llm", "probability": pred["failure_probability"] * 1.1},
        ]

    return response


@app.get(
    "/api/v1/predictions",
    response_model=Dict[str, PredictionResponse],
    tags=["Predictions"],
)
async def get_all_predictions(
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    api_key: ApiKey = Depends(require_permission(Permission.READ_PREDICTIONS)),
):
    """Get failure predictions for all miners"""
    auth = get_auth_manager()
    auth.record_request(api_key, "/api/v1/predictions")

    result = {}
    for miner_id, pred in data_store.predictions.items():
        if risk_level and pred["risk_level"] != risk_level:
            continue

        result[miner_id] = PredictionResponse(
            miner_id=miner_id,
            timestamp=pred["timestamp"],
            failure_probability=pred["failure_probability"],
            confidence=pred["confidence"],
            risk_level=pred["risk_level"],
            recommended_action=pred["recommended_action"],
            reasoning=pred["reasoning"],
        )

    return result


# ============ 12. ROUTING ENDPOINTS ============

@app.get("/api/v1/routing", response_model=RoutingTableResponse, tags=["Routing"])
async def get_routing_table(
    api_key: ApiKey = Depends(require_permission(Permission.READ_ROUTING)),
):
    """Get current routing table"""
    auth = get_auth_manager()
    auth.record_request(api_key, "/api/v1/routing")

    healthy_count = sum(1 for m in data_store.miners.values() if m["is_healthy"])

    return RoutingTableResponse(
        timestamp=time.time(),
        total_miners=len(data_store.miners),
        healthy_miners=healthy_count,
        routing_strategy="weighted_round_robin",
        miners={
            miner_id: {
                "weight": data["routing_weight"],
                "status": "active" if data["is_healthy"] else "degraded",
                "health_score": data["health_score"],
            }
            for miner_id, data in data_store.miners.items()
        },
    )


@app.get("/api/v1/routing/weights", tags=["Routing"])
async def get_routing_weights(
    api_key: ApiKey = Depends(require_permission(Permission.READ_ROUTING)),
):
    """Get current routing weights for all miners"""
    auth = get_auth_manager()
    auth.record_request(api_key, "/api/v1/routing/weights")

    return {
        miner_id: data["routing_weight"] for miner_id, data in data_store.miners.items()
    }


@app.get("/api/v1/decisions", response_model=List[DecisionResponse], tags=["Routing"])
async def get_recent_decisions(
    limit: int = Query(10, ge=1, le=100),
    api_key: ApiKey = Depends(require_permission(Permission.READ_ROUTING)),
):
    """Get recent routing decisions"""
    auth = get_auth_manager()
    auth.record_request(api_key, "/api/v1/decisions")

    # Generate sample decisions
    decisions = []
    for miner_id, data in list(data_store.miners.items())[:limit]:
        decisions.append(
            DecisionResponse(
                miner_id=miner_id,
                timestamp=time.time() - 300,
                old_weight=1.0,
                new_weight=data["routing_weight"],
                reason=f"Health score: {data['health_score']:.0f}%",
                applied=True,
            )
        )

    return decisions


# ============ 13. ADMIN ENDPOINTS ============

@app.post("/api/v1/admin/keys", response_model=ApiKeyResponse, tags=["Admin"])
async def create_api_key(
    request: ApiKeyRequest,
    api_key: ApiKey = Depends(require_permission(Permission.ADMIN)),
):
    """Create a new API key (admin only)"""
    auth = get_auth_manager()
    auth.record_request(api_key, "/api/v1/admin/keys")

    try:
        tier = AccessTier(request.tier)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {request.tier}")

    key_id, raw_key = auth.create_api_key(
        name=request.name,
        tier=tier,
        expires_in_days=request.expires_in_days,
        owner_email=request.owner_email,
    )

    return ApiKeyResponse(
        key_id=key_id,
        api_key=raw_key,
        tier=request.tier,
        message="API key created successfully. Save this key - it won't be shown again!",
    )


@app.get("/api/v1/admin/keys", tags=["Admin"])
async def list_api_keys(
    include_inactive: bool = Query(False),
    api_key: ApiKey = Depends(require_permission(Permission.ADMIN)),
):
    """List all API keys (admin only)"""
    auth = get_auth_manager()
    auth.record_request(api_key, "/api/v1/admin/keys")

    return auth.list_keys(include_inactive=include_inactive)


@app.delete("/api/v1/admin/keys/{key_id}", tags=["Admin"])
async def revoke_api_key(
    key_id: str, api_key: ApiKey = Depends(require_permission(Permission.ADMIN))
):
    """Revoke an API key (admin only)"""
    auth = get_auth_manager()
    auth.record_request(api_key, f"/api/v1/admin/keys/{key_id}")

    if auth.revoke_key(key_id):
        return {"message": f"API key {key_id} revoked"}
    else:
        raise HTTPException(status_code=404, detail=f"Key not found: {key_id}")


@app.get("/api/v1/admin/usage/{key_id}", tags=["Admin"])
async def get_key_usage(
    key_id: str, api_key: ApiKey = Depends(require_permission(Permission.ADMIN))
):
    """Get usage statistics for an API key (admin only)"""
    auth = get_auth_manager()
    auth.record_request(api_key, f"/api/v1/admin/usage/{key_id}")

    stats = auth.get_usage_stats(key_id)
    if not stats:
        raise HTTPException(status_code=404, detail=f"Key not found: {key_id}")

    return stats


# ============ 14. ERROR HANDLERS ============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            retry_after=exc.headers.get("Retry-After") if exc.headers else None,
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal server error", detail=str(exc)).dict(),
    )


# ============ 15. RUN SERVER ============

def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api()