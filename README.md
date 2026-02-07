# Predictive Router Oracle

An AI-driven routing oracle that predicts miner/validator failures before they happen and actively redirects traffic to healthier nodes to prevent outages.

## Overview

The Predictive Router Oracle is an AI Ops Agent for the Cortensor network that:

- **Continuously monitors** miner/validator health metrics
- **Detects early warning signals**: latency spikes, missed heartbeats, reduced throughput, error rates
- **Predicts failures** using multi-model AI (Statistical + ML + LLM)
- **Arbitrates predictions** via Agent Arbitration Nexus (consensus-based)
- **Controls routing** dynamically via off-chain load balancer + on-chain registry
- **Publishes** routing decisions, incident reports, and prediction confidence
- **Exposes** a paid Prediction API for external consumers

**The key idea**: Don't wait for nodes to die — route around them before they fail.

## Architecture

```
┌───────────────┐
│ Miner Metrics │
└───────┬───────┘
        ↓
┌────────────────────┐
│ Metrics Collector  │ ← Latency, throughput, errors, heartbeats
└───────┬────────────┘
        ↓
┌────────────────────────────┐
│ Prediction Agents          │
│ - Statistical (Isolation   │
│   Forest anomaly detection)│
│ - ML (Gradient Boosting    │
│   failure classifier)      │
│ - LLM (GPT-4 reasoning)    │
└───────┬────────────────────┘
        ↓
┌────────────────────────────┐
│ Agent Arbitration Nexus    │ ← Consensus-based decision making
└───────┬────────────────────┘
        ↓
┌────────────────────────────┐
│ Routing Decision Engine    │ ← Traffic Controller
└───────┬────────────────────┘
        ↓
┌────────────────────────────┐
│ Router Oracle              │
│ - Off-chain load balancer  │
│ - On-chain registry        │
└────────────────────────────┘
```

## Features

### Multi-Model Prediction
- **Statistical Agent**: Isolation Forest for anomaly detection, trend analysis
- **ML Agent**: Gradient Boosting Classifier for failure prediction
- **LLM Agent**: GPT-4/Claude for explainable reasoning

### Agent Arbitration Nexus
- Consensus-based decision making
- Outlier detection and rejection
- Weighted voting by model confidence
- Fallback mechanisms for disagreement

### Hybrid Routing Control
- **Off-chain**: Fast weighted round-robin load balancer
- **On-chain**: Solidity contract for auditability and trust

### Paid Prediction API
- Rate-limited API access
- Multiple tiers (Free, Basic, Pro, Enterprise)
- Real-time predictions and routing data

## Project Structure

```
predictive-router-oracle/
├── agents/                 # AI Prediction Agents
│   ├── latency_analyzer.py     # Statistical anomaly detection
│   ├── failure_predictor.py    # ML classification
│   ├── llm_reasoner.py         # LLM-based reasoning
│   └── arbitration_nexus.py    # Meta-agent consensus
├── metrics/                # Metrics Collection
│   ├── collector.py            # Central metrics collector
│   ├── heartbeat_monitor.py    # Heartbeat tracking
│   └── miner_profiler.py       # Miner behavior profiling
├── router/                 # Routing Layer
│   ├── router_service.py       # Load balancer
│   ├── routing_table.json      # Current routing state
│   └── traffic_controller.py   # Orchestration
├── oracle/                 # Oracle Layer
│   ├── oracle_publisher.py     # Multi-target publisher
│   ├── onchain_registry.sol    # Smart contract
│   └── oracle_adapter.py       # System adapter
├── api/                    # Prediction API
│   ├── prediction_api.py       # FastAPI endpoints
│   └── auth.py                 # Authentication
├── reports/                # Incident Reporting
│   ├── incident_generator.py   # Report generation
│   └── templates/              # HTML templates
├── dashboard/              # Monitoring Dashboard
│   ├── backend/                # WebSocket server
│   └── frontend/               # React dashboard
├── config/                 # Configuration
│   ├── thresholds.yaml         # Alert thresholds
│   └── models.yaml             # Model parameters
├── scripts/                # Utilities
│   ├── simulate_failures.py    # Failure simulation
│   └── demo_runner.py          # Demo orchestration
├── docker-compose.yml      # Docker orchestration
└── README.md
```

## Quick Start

### 1. Run the Demo

```bash
python scripts/demo_runner.py --mode demo
```

This runs a complete demonstration of the prediction and routing system.

### 2. Start with Docker

```bash
# Copy environment file
cp .env.example .env

# Edit .env with your API keys
nano .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 3. Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8080

## API Usage

### Get Predictions

```bash
# Get all miners health
curl -H "X-API-Key: dev_xxx" http://localhost:8000/api/v1/miners

# Get specific prediction
curl -H "X-API-Key: dev_xxx" http://localhost:8000/api/v1/predictions/miner_001

# Get routing table
curl -H "X-API-Key: dev_xxx" http://localhost:8000/api/v1/routing
```

### API Tiers

| Tier | Rate Limit | Features |
|------|------------|----------|
| Free | 10/min | Health data only |
| Basic | 60/min | + Predictions |
| Pro | 300/min | + Routing data |
| Enterprise | 1000/min | + Admin access |

## Smart Contract

Deploy `oracle/onchain_registry.sol` to your preferred EVM chain:

```solidity
// Key functions
function updateMinerHealth(address miner, uint256 healthScore, uint256 failureProbability, uint256 routingWeight, bytes32 evidenceHash)
function emergencyReroute(address miner, string reason)
function queryMinerHealth(address miner) payable returns (...)
function getEligibleMiners() returns (address[])
```

## Configuration

### Thresholds (config/thresholds.yaml)

```yaml
latency:
  warning: 200ms
  critical: 500ms

prediction:
  reroute_threshold: 0.7
  warning_threshold: 0.5

arbitration:
  agreement_threshold: 2
  ml_weight: 0.4
  statistical_weight: 0.3
  llm_weight: 0.3
```

### Models (config/models.yaml)

```yaml
statistical_model:
  name: IsolationForest
  contamination: 0.1

ml_model:
  name: GradientBoostingClassifier
  n_estimators: 200

llm_agent:
  provider: openai
  model: gpt-4
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for LLM agent |
| `ETH_RPC_URL` | Ethereum RPC endpoint |
| `CONTRACT_ADDRESS` | Deployed oracle contract |
| `ORACLE_PRIVATE_KEY` | Wallet for signing transactions |
| `CONTROLLER_MODE` | automatic/supervised/manual |

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Simulate Failures

```bash
# Interactive mode
python scripts/simulate_failures.py --interactive

# Run specific scenario
python scripts/simulate_failures.py --scenario cascade

# Continuous simulation
python scripts/simulate_failures.py --continuous
```

## Monitoring

The dashboard provides real-time visibility into:

- Miner health status (🟢 Healthy, 🟡 Degraded, 🔴 Critical)
- Failure probabilities and predictions
- Routing weight distribution
- Recent decisions and events
- AI prediction confidence

## Security Considerations

- Never commit `.env` or private keys
- Use hardware wallets for production oracle keys
- Implement rate limiting on all endpoints
- Validate all incoming miner data
- Use TLS in production

## License

MIT License - See LICENSE file for details.

---

Built for the Cortensor Network