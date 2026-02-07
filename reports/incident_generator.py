"""
Incident Generator Module
Generates incident reports and summary reports
"""

import time
import logging
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from jinja2 import Environment, FileSystemLoader, select_autoescape
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status"""

    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IncidentType(Enum):
    """Types of incidents"""

    LATENCY_SPIKE = "latency_spike"
    HIGH_ERROR_RATE = "high_error_rate"
    NODE_FAILURE = "node_failure"
    HEARTBEAT_LOSS = "heartbeat_loss"
    PREDICTION_TRIGGERED = "prediction_triggered"
    EMERGENCY_REROUTE = "emergency_reroute"


@dataclass
class TimelineEvent:
    """Event in incident timeline"""

    timestamp: float
    description: str
    event_type: str


@dataclass
class Incident:
    """Incident record"""

    incident_id: str
    miner_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: float
    title: str
    description: str

    # Timeline
    timeline: List[TimelineEvent] = field(default_factory=list)

    # Metrics at time of incident
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)

    # AI prediction data
    predictions: List[Dict[str, Any]] = field(default_factory=list)

    # Actions taken
    actions: List[str] = field(default_factory=list)
    recommendation: str = ""
    root_cause: str = ""

    # Resolution
    resolved_at: Optional[float] = None
    resolution_notes: str = ""

    # Impact
    affected_requests: int = 0
    downtime_seconds: float = 0


class IncidentGenerator:
    """
    Generates incident reports from system events.
    Produces HTML and JSON reports.
    """

    def __init__(self, templates_dir: str = "reports/templates"):
        self.templates_dir = templates_dir
        self.incidents: Dict[str, Incident] = {}
        self.output_dir = "reports/output"

        # Initialize Jinja2 environment
        self._init_jinja()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_jinja(self):
        """Initialize Jinja2 template environment"""
        try:
            self.jinja_env = Environment(
                loader=FileSystemLoader(self.templates_dir),
                autoescape=select_autoescape(["html", "xml"]),
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Jinja2: {e}")
            self.jinja_env = None

    def _generate_incident_id(self, miner_id: str) -> str:
        """Generate unique incident ID"""
        timestamp = str(time.time())
        hash_input = f"{miner_id}_{timestamp}"
        hash_output = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"INC-{hash_output.upper()}"

    def _format_timestamp(self, ts: float) -> str:
        """Format timestamp for display"""
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _format_duration(self, seconds: float) -> str:
        """Format duration for display"""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            return f"{seconds / 60:.1f} minutes"
        else:
            return f"{seconds / 3600:.1f} hours"

    def create_incident(
        self,
        miner_id: str,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        title: str,
        description: str,
        metrics_snapshot: Optional[Dict[str, Any]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
    ) -> Incident:
        """Create a new incident"""
        incident_id = self._generate_incident_id(miner_id)

        incident = Incident(
            incident_id=incident_id,
            miner_id=miner_id,
            incident_type=incident_type,
            severity=severity,
            status=IncidentStatus.OPEN,
            created_at=time.time(),
            title=title,
            description=description,
            metrics_snapshot=metrics_snapshot or {},
            predictions=predictions or [],
        )

        # Add initial timeline event
        incident.timeline.append(
            TimelineEvent(
                timestamp=time.time(),
                description=f"Incident created: {title}",
                event_type="created",
            )
        )

        self.incidents[incident_id] = incident
        logger.info(f"Created incident: {incident_id} for {miner_id}")

        return incident

    def add_timeline_event(
        self, incident_id: str, description: str, event_type: str = "update"
    ) -> None:
        """Add event to incident timeline"""
        if incident_id not in self.incidents:
            return

        self.incidents[incident_id].timeline.append(
            TimelineEvent(
                timestamp=time.time(), description=description, event_type=event_type
            )
        )

    def add_action(self, incident_id: str, action: str) -> None:
        """Record an action taken for the incident"""
        if incident_id not in self.incidents:
            return

        self.incidents[incident_id].actions.append(action)
        self.add_timeline_event(incident_id, f"Action: {action}", "action")

    def update_status(self, incident_id: str, status: IncidentStatus) -> None:
        """Update incident status"""
        if incident_id not in self.incidents:
            return

        incident = self.incidents[incident_id]
        old_status = incident.status
        incident.status = status

        self.add_timeline_event(
            incident_id,
            f"Status changed: {old_status.value} -> {status.value}",
            "status_change",
        )

        if status == IncidentStatus.RESOLVED:
            incident.resolved_at = time.time()
            incident.downtime_seconds = incident.resolved_at - incident.created_at

    def set_root_cause(self, incident_id: str, root_cause: str) -> None:
        """Set root cause analysis"""
        if incident_id in self.incidents:
            self.incidents[incident_id].root_cause = root_cause

    def set_recommendation(self, incident_id: str, recommendation: str) -> None:
        """Set recommendation"""
        if incident_id in self.incidents:
            self.incidents[incident_id].recommendation = recommendation

    def close_incident(self, incident_id: str, resolution_notes: str = "") -> None:
        """Close an incident"""
        if incident_id not in self.incidents:
            return

        incident = self.incidents[incident_id]
        incident.status = IncidentStatus.CLOSED
        incident.resolution_notes = resolution_notes

        if not incident.resolved_at:
            incident.resolved_at = time.time()
            incident.downtime_seconds = incident.resolved_at - incident.created_at

        self.add_timeline_event(incident_id, "Incident closed", "closed")

    def generate_incident_report(
        self, incident_id: str, output_format: str = "html"
    ) -> Optional[str]:
        """
        Generate incident report.

        Args:
            incident_id: ID of the incident
            output_format: 'html' or 'json'

        Returns:
            Path to generated report or None
        """
        if incident_id not in self.incidents:
            return None

        incident = self.incidents[incident_id]

        if output_format == "json":
            return self._generate_json_report(incident)
        else:
            return self._generate_html_report(incident)

    def _generate_html_report(self, incident: Incident) -> Optional[str]:
        """Generate HTML incident report"""
        if not self.jinja_env:
            logger.error("Jinja2 not initialized")
            return None

        try:
            template = self.jinja_env.get_template("incident_report.html")

            # Prepare template data
            data = {
                "incident_id": incident.incident_id,
                "title": incident.title,
                "miner_id": incident.miner_id,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "detection_time": self._format_timestamp(incident.created_at),
                "duration": self._format_duration(incident.downtime_seconds),
                "description": incident.description,
                "timeline": [
                    {
                        "time": self._format_timestamp(e.timestamp),
                        "description": e.description,
                    }
                    for e in incident.timeline
                ],
                "metrics": [
                    {
                        "name": k,
                        "value": f"{v:.2f}" if isinstance(v, float) else str(v),
                        "threshold": "N/A",
                        "status": "Normal",
                    }
                    for k, v in incident.metrics_snapshot.items()
                ],
                "predictions": incident.predictions,
                "actions": incident.actions,
                "recommendation": incident.recommendation,
                "root_cause": incident.root_cause or "Under investigation",
                "generated_at": self._format_timestamp(time.time()),
            }

            html_content = template.render(**data)

            # Save to file
            output_path = os.path.join(
                self.output_dir, f"incident_{incident.incident_id}.html"
            )

            with open(output_path, "w") as f:
                f.write(html_content)

            logger.info(f"Generated HTML report: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return None

    def _generate_json_report(self, incident: Incident) -> Optional[str]:
        """Generate JSON incident report"""
        try:
            data = {
                "incident_id": incident.incident_id,
                "miner_id": incident.miner_id,
                "incident_type": incident.incident_type.value,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "created_at": incident.created_at,
                "resolved_at": incident.resolved_at,
                "title": incident.title,
                "description": incident.description,
                "timeline": [
                    {
                        "timestamp": e.timestamp,
                        "description": e.description,
                        "event_type": e.event_type,
                    }
                    for e in incident.timeline
                ],
                "metrics_snapshot": incident.metrics_snapshot,
                "predictions": incident.predictions,
                "actions": incident.actions,
                "recommendation": incident.recommendation,
                "root_cause": incident.root_cause,
                "resolution_notes": incident.resolution_notes,
                "affected_requests": incident.affected_requests,
                "downtime_seconds": incident.downtime_seconds,
                "generated_at": time.time(),
            }

            output_path = os.path.join(
                self.output_dir, f"incident_{incident.incident_id}.json"
            )

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Generated JSON report: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            return None

    def generate_summary_report(
        self,
        period: str = "Last 24 Hours",
        miners_data: Optional[List[Dict[str, Any]]] = None,
        output_format: str = "html",
    ) -> Optional[str]:
        """Generate summary report for all miners"""
        if not self.jinja_env and output_format == "html":
            logger.error("Jinja2 not initialized")
            return None

        # Gather summary data
        all_incidents = list(self.incidents.values())

        # Use provided miners data or generate sample
        if not miners_data:
            miners_data = [
                {
                    "id": "miner_001",
                    "status": "healthy",
                    "health_score": 95,
                    "failure_probability": 8,
                    "risk_level": "low",
                    "routing_weight": 0.92,
                },
                {
                    "id": "miner_002",
                    "status": "degraded",
                    "health_score": 72,
                    "failure_probability": 35,
                    "risk_level": "medium",
                    "routing_weight": 0.65,
                },
                {
                    "id": "miner_003",
                    "status": "healthy",
                    "health_score": 88,
                    "failure_probability": 15,
                    "risk_level": "low",
                    "routing_weight": 0.85,
                },
            ]

        summary_data = {
            "period": period,
            "total_miners": len(miners_data),
            "healthy_miners": sum(
                1 for m in miners_data if m.get("status") == "healthy"
            ),
            "healthy_percentage": int(
                sum(1 for m in miners_data if m.get("status") == "healthy")
                / len(miners_data)
                * 100
            )
            if miners_data
            else 0,
            "total_incidents": len(all_incidents),
            "total_reroutes": sum(
                1
                for i in all_incidents
                if i.incident_type == IncidentType.EMERGENCY_REROUTE
            ),
            "miners": miners_data,
            "incidents": [
                {
                    "time": self._format_timestamp(i.created_at),
                    "miner": i.miner_id,
                    "type": i.incident_type.value,
                    "severity": i.severity.value,
                    "resolution": i.status.value,
                }
                for i in all_incidents[:10]  # Last 10 incidents
            ],
            "generated_at": self._format_timestamp(time.time()),
        }

        if output_format == "json":
            output_path = os.path.join(self.output_dir, "summary_report.json")
            with open(output_path, "w") as f:
                json.dump(summary_data, f, indent=2)
            return output_path

        try:
            template = self.jinja_env.get_template("summary_report.html")
            html_content = template.render(**summary_data)

            output_path = os.path.join(self.output_dir, "summary_report.html")
            with open(output_path, "w") as f:
                f.write(html_content)

            logger.info(f"Generated summary report: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return None

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID"""
        return self.incidents.get(incident_id)

    def get_open_incidents(self) -> List[Incident]:
        """Get all open incidents"""
        return [
            i
            for i in self.incidents.values()
            if i.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]
        ]

    def get_incidents_by_miner(self, miner_id: str) -> List[Incident]:
        """Get all incidents for a specific miner"""
        return [i for i in self.incidents.values() if i.miner_id == miner_id]

    def export_all_incidents(self) -> str:
        """Export all incidents to JSON"""
        output_path = os.path.join(self.output_dir, "all_incidents.json")

        data = {
            "exported_at": time.time(),
            "total_incidents": len(self.incidents),
            "incidents": [
                {
                    "incident_id": i.incident_id,
                    "miner_id": i.miner_id,
                    "type": i.incident_type.value,
                    "severity": i.severity.value,
                    "status": i.status.value,
                    "created_at": i.created_at,
                    "resolved_at": i.resolved_at,
                    "title": i.title,
                    "downtime_seconds": i.downtime_seconds,
                }
                for i in self.incidents.values()
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path


# Singleton instance
_generator_instance: Optional[IncidentGenerator] = None


def get_incident_generator() -> IncidentGenerator:
    """Get or create the global incident generator instance"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = IncidentGenerator()
    return _generator_instance