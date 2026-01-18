"""IDS-Agent EV charging session classifier."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple


@dataclass(frozen=True)
class SessionFeatures:
    """Normalized EV charging session inputs."""

    connection_time: datetime
    disconnect_time: datetime
    requested_demand: float
    kwh_delivered: float

    @property
    def idle_time(self) -> float:
        """Idle time after charging in minutes (disconnect - connection)."""
        delta = self.disconnect_time - self.connection_time
        return max(0.0, delta.total_seconds() / 60.0)

    @property
    def efficiency(self) -> float:
        """Charging efficiency (delivered / requested)."""
        if self.requested_demand <= 0:
            return 0.0
        return max(0.0, self.kwh_delivered / self.requested_demand)

    @property
    def demand_mismatch(self) -> float:
        """Mismatch between requested demand and delivered energy."""
        if self.requested_demand <= 0:
            return 1.0
        return max(0.0, (self.requested_demand - self.kwh_delivered) / self.requested_demand)


@dataclass(frozen=True)
class DetectionResult:
    """IDS-Agent decision and explanation."""

    prediction: str
    explanation: str
    indicators: Dict[str, float]

    def render(self) -> str:
        return f"{self.explanation}\nPrediction: {self.prediction}"


def classify_session(features: SessionFeatures) -> DetectionResult:
    """Classify a charging session using heuristic indicators.

    Thresholds are conservative defaults and should be calibrated using
    labeled data once available.
    """

    efficiency = features.efficiency
    idle_time = features.idle_time
    mismatch = features.demand_mismatch

    low_efficiency = efficiency < 0.2
    long_idle = idle_time > 30
    large_mismatch = mismatch > 0.5

    triggers = []
    if low_efficiency:
        triggers.append("very low charging efficiency")
    if long_idle:
        triggers.append("large idle time after charging")
    if large_mismatch:
        triggers.append("large mismatch between requested demand and delivered energy")

    prediction = "Attack" if triggers else "Normal"
    if triggers:
        explanation = (
            "Potential attack indicators detected: "
            + ", ".join(triggers)
            + "."
        )
    else:
        explanation = "No strong attack indicators detected in efficiency, idle time, or demand match."

    return DetectionResult(
        prediction=prediction,
        explanation=explanation,
        indicators={
            "efficiency": round(efficiency, 4),
            "idle_time": round(idle_time, 2),
            "demand_mismatch": round(mismatch, 4),
        },
    )


def _parse_timestamp(value: float | str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value))
    return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S%z")


def parse_features(payload: Dict[str, float | str]) -> SessionFeatures:
    """Parse raw payload into SessionFeatures."""
    return SessionFeatures(
        connection_time=_parse_timestamp(payload["connectionTime"]),
        disconnect_time=_parse_timestamp(payload["disconnectTime"]),
        requested_demand=float(payload["RequestedDemand"]),
        kwh_delivered=float(payload["kWhDelivered"]),
    )


def explain_prediction(payload: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    """Return the human-readable response and indicator metrics."""
    result = classify_session(parse_features(payload))
    return result.render(), result.indicators
