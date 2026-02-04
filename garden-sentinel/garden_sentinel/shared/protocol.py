"""
Shared protocol definitions for Garden Sentinel system.
Defines message types for communication between edge devices and server.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import json


class MessageType(Enum):
    # Edge -> Server
    FRAME = "frame"
    HEARTBEAT = "heartbeat"
    STATUS = "status"
    DETECTION_RESULT = "detection_result"  # For edge inference

    # Server -> Edge
    COMMAND = "command"
    CONFIG_UPDATE = "config_update"
    DETECTION_ALERT = "detection_alert"


class CommandType(Enum):
    ACTIVATE_ALARM = "activate_alarm"
    DEACTIVATE_ALARM = "deactivate_alarm"
    ACTIVATE_SPRAYER = "activate_sprayer"
    DEACTIVATE_SPRAYER = "deactivate_sprayer"
    CAPTURE_SNAPSHOT = "capture_snapshot"
    UPDATE_CONFIG = "update_config"
    REBOOT = "reboot"


class PredatorType(Enum):
    FOX = "fox"
    BADGER = "badger"
    CAT = "cat"
    DOG = "dog"
    HAWK = "hawk"
    EAGLE = "eagle"
    OWL = "owl"
    CROW = "crow"
    MAGPIE = "magpie"
    RAT = "rat"
    WEASEL = "weasel"
    STOAT = "stoat"
    MINK = "mink"
    UNKNOWN_PREDATOR = "unknown_predator"


class ThreatLevel(Enum):
    LOW = "low"          # Monitor only
    MEDIUM = "medium"    # Alert user
    HIGH = "high"        # Alert + alarm
    CRITICAL = "critical"  # Alert + alarm + sprayer


# Threat level mapping for predators
PREDATOR_THREAT_LEVELS = {
    PredatorType.FOX: ThreatLevel.CRITICAL,
    PredatorType.BADGER: ThreatLevel.HIGH,
    PredatorType.CAT: ThreatLevel.MEDIUM,
    PredatorType.DOG: ThreatLevel.HIGH,
    PredatorType.HAWK: ThreatLevel.CRITICAL,
    PredatorType.EAGLE: ThreatLevel.CRITICAL,
    PredatorType.OWL: ThreatLevel.HIGH,
    PredatorType.CROW: ThreatLevel.MEDIUM,
    PredatorType.MAGPIE: ThreatLevel.MEDIUM,
    PredatorType.RAT: ThreatLevel.MEDIUM,
    PredatorType.WEASEL: ThreatLevel.CRITICAL,
    PredatorType.STOAT: ThreatLevel.CRITICAL,
    PredatorType.MINK: ThreatLevel.CRITICAL,
    PredatorType.UNKNOWN_PREDATOR: ThreatLevel.HIGH,
}


@dataclass
class BoundingBox:
    x: float  # Top-left x (normalized 0-1)
    y: float  # Top-left y (normalized 0-1)
    width: float  # Width (normalized 0-1)
    height: float  # Height (normalized 0-1)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict) -> "BoundingBox":
        return cls(x=data["x"], y=data["y"], width=data["width"], height=data["height"])


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: BoundingBox
    predator_type: Optional[PredatorType] = None
    threat_level: Optional[ThreatLevel] = None

    def __post_init__(self):
        # Try to map to predator type
        if self.predator_type is None:
            try:
                self.predator_type = PredatorType(self.class_name.lower())
                self.threat_level = PREDATOR_THREAT_LEVELS.get(
                    self.predator_type, ThreatLevel.MEDIUM
                )
            except ValueError:
                pass

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "predator_type": self.predator_type.value if self.predator_type else None,
            "threat_level": self.threat_level.value if self.threat_level else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Detection":
        return cls(
            class_name=data["class_name"],
            confidence=data["confidence"],
            bbox=BoundingBox.from_dict(data["bbox"]),
            predator_type=PredatorType(data["predator_type"]) if data.get("predator_type") else None,
            threat_level=ThreatLevel(data["threat_level"]) if data.get("threat_level") else None,
        )


@dataclass
class EdgeMessage:
    device_id: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "device_id": self.device_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        })

    @classmethod
    def from_json(cls, data: str) -> "EdgeMessage":
        d = json.loads(data)
        return cls(
            device_id=d["device_id"],
            message_type=MessageType(d["message_type"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            payload=d.get("payload", {}),
        )


@dataclass
class ServerCommand:
    target_device: str  # Device ID or "*" for broadcast
    command_type: CommandType
    parameters: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_json(self) -> str:
        return json.dumps({
            "target_device": self.target_device,
            "command_type": self.command_type.value,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
        })

    @classmethod
    def from_json(cls, data: str) -> "ServerCommand":
        d = json.loads(data)
        return cls(
            target_device=d["target_device"],
            command_type=CommandType(d["command_type"]),
            parameters=d.get("parameters", {}),
            timestamp=datetime.fromisoformat(d["timestamp"]),
        )


@dataclass
class AlertEvent:
    event_id: str
    device_id: str
    timestamp: datetime
    detections: list[Detection]
    max_threat_level: ThreatLevel
    frame_path: Optional[str] = None
    actions_taken: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "detections": [d.to_dict() for d in self.detections],
            "max_threat_level": self.max_threat_level.value,
            "frame_path": self.frame_path,
            "actions_taken": self.actions_taken,
        }
