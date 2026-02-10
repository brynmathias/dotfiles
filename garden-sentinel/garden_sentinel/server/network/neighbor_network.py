"""
Neighbor network for sharing predator alerts between nearby users.

Allows Garden Sentinel users to share sightings and receive
early warnings when predators are heading their direction.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Set
from enum import Enum
import asyncio
import json
import logging
import time
import math
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust level for neighbors."""
    PENDING = "pending"      # Awaiting approval
    BASIC = "basic"          # Can receive alerts
    TRUSTED = "trusted"      # Can receive and send alerts
    VERIFIED = "verified"    # Verified neighbor (met in person, etc.)
    BLOCKED = "blocked"      # Blocked user


@dataclass
class NetworkConfig:
    """Configuration for neighbor network."""
    enabled: bool = True
    auto_share_alerts: bool = True
    share_radius_km: float = 2.0  # Share with neighbors within this radius
    receive_radius_km: float = 5.0  # Receive alerts from this radius
    min_confidence_to_share: float = 0.7
    share_predator_types: List[str] = field(default_factory=lambda: ["fox", "coyote", "hawk"])
    anonymous_mode: bool = False  # Hide exact location
    location_blur_meters: float = 100.0  # Blur location by this amount


@dataclass
class Neighbor:
    """A neighbor in the network."""
    id: str
    name: str
    site_id: Optional[str] = None
    location: Optional[Dict[str, float]] = None  # lat, lng
    distance_km: Optional[float] = None
    trust_level: TrustLevel = TrustLevel.PENDING
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    alerts_received: int = 0
    alerts_sent: int = 0
    public_key: Optional[str] = None  # For encrypted communication

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "site_id": self.site_id,
            "location": self.location,
            "distance_km": self.distance_km,
            "trust_level": self.trust_level.value,
            "connected_at": self.connected_at,
            "last_seen": self.last_seen,
            "alerts_received": self.alerts_received,
            "alerts_sent": self.alerts_sent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Neighbor":
        return cls(
            id=data["id"],
            name=data["name"],
            site_id=data.get("site_id"),
            location=data.get("location"),
            distance_km=data.get("distance_km"),
            trust_level=TrustLevel(data.get("trust_level", "pending")),
            connected_at=data.get("connected_at", time.time()),
            last_seen=data.get("last_seen", time.time()),
            alerts_received=data.get("alerts_received", 0),
            alerts_sent=data.get("alerts_sent", 0),
            public_key=data.get("public_key"),
        )


@dataclass
class SharedAlert:
    """An alert shared over the neighbor network."""
    id: str
    source_id: str  # Neighbor ID who sent it (or "self")
    predator_type: str
    confidence: float
    timestamp: float
    location: Dict[str, float]  # lat, lng (possibly blurred)
    heading: Optional[float] = None  # Direction of travel (degrees)
    speed_estimate: Optional[float] = None  # Estimated speed m/s
    description: Optional[str] = None
    expires_at: float = field(default_factory=lambda: time.time() + 1800)  # 30 min default
    hops: int = 0  # Number of times relayed
    verified: bool = False

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def estimated_arrival_minutes(self, my_location: Dict[str, float]) -> Optional[float]:
        """Estimate minutes until predator could arrive at location."""
        if not self.speed_estimate or self.speed_estimate <= 0:
            return None

        distance = self._haversine(
            self.location["lat"], self.location["lng"],
            my_location["lat"], my_location["lng"]
        )
        return (distance / self.speed_estimate) / 60

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in meters between two points."""
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "predator_type": self.predator_type,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "location": self.location,
            "heading": self.heading,
            "speed_estimate": self.speed_estimate,
            "description": self.description,
            "expires_at": self.expires_at,
            "hops": self.hops,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedAlert":
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            predator_type=data["predator_type"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            location=data["location"],
            heading=data.get("heading"),
            speed_estimate=data.get("speed_estimate"),
            description=data.get("description"),
            expires_at=data.get("expires_at", time.time() + 1800),
            hops=data.get("hops", 0),
            verified=data.get("verified", False),
        )


# Callback types
AlertCallback = Callable[[SharedAlert], Any]


class NeighborNetwork:
    """
    Peer-to-peer network for sharing predator alerts.

    Features:
    - Discover nearby Garden Sentinel users
    - Share and receive predator alerts
    - Trust-based permissions
    - Location privacy (blurring)
    - Alert relay for extended range
    """

    MAX_HOPS = 3  # Maximum relay hops
    DISCOVERY_INTERVAL = 300  # 5 minutes
    CLEANUP_INTERVAL = 600  # 10 minutes

    def __init__(
        self,
        my_id: str,
        my_name: str,
        my_location: Dict[str, float],
        config: Optional[NetworkConfig] = None,
        storage_path: Optional[Path] = None,
        server_url: Optional[str] = None,
    ):
        self.my_id = my_id
        self.my_name = my_name
        self.my_location = my_location
        self.config = config or NetworkConfig()
        self.storage_path = storage_path
        self.server_url = server_url  # Central relay server (optional)

        self.neighbors: Dict[str, Neighbor] = {}
        self.received_alerts: Dict[str, SharedAlert] = {}
        self.sent_alert_ids: Set[str] = set()

        self._alert_callbacks: List[AlertCallback] = []
        self._running = False
        self._tasks: List[asyncio.Task] = []

        if storage_path:
            self._load_data()

    def _load_data(self):
        """Load neighbors from storage."""
        if not self.storage_path:
            return

        neighbors_file = self.storage_path / "neighbors.json"
        if neighbors_file.exists():
            with open(neighbors_file) as f:
                data = json.load(f)
                for n in data:
                    neighbor = Neighbor.from_dict(n)
                    self.neighbors[neighbor.id] = neighbor
            logger.info(f"Loaded {len(self.neighbors)} neighbors")

    def _save_data(self):
        """Save neighbors to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        neighbors_file = self.storage_path / "neighbors.json"
        with open(neighbors_file, "w") as f:
            json.dump([n.to_dict() for n in self.neighbors.values()], f, indent=2)

    def add_alert_callback(self, callback: AlertCallback):
        """Add callback for incoming alerts."""
        self._alert_callbacks.append(callback)

    async def start(self):
        """Start the neighbor network."""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._discovery_loop()))
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))

        if self.server_url:
            self._tasks.append(asyncio.create_task(self._server_connection_loop()))

        logger.info("Neighbor network started")

    async def stop(self):
        """Stop the neighbor network."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        logger.info("Neighbor network stopped")

    async def _discovery_loop(self):
        """Periodically discover new neighbors."""
        while self._running:
            try:
                await self._discover_neighbors()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Discovery error: {e}")
            await asyncio.sleep(self.DISCOVERY_INTERVAL)

    async def _cleanup_loop(self):
        """Periodically clean up expired alerts and stale neighbors."""
        while self._running:
            try:
                self._cleanup_expired_alerts()
                self._cleanup_stale_neighbors()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            await asyncio.sleep(self.CLEANUP_INTERVAL)

    async def _server_connection_loop(self):
        """Maintain connection to relay server."""
        while self._running:
            try:
                await self._connect_to_server()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Server connection error: {e}")
            await asyncio.sleep(30)

    async def _discover_neighbors(self):
        """Discover nearby neighbors."""
        if not self.server_url:
            return

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Register/update our presence
                payload = {
                    "id": self.my_id,
                    "name": self.my_name if not self.config.anonymous_mode else "Anonymous",
                    "location": self._blur_location(self.my_location),
                    "radius_km": self.config.receive_radius_km,
                }

                async with session.post(
                    f"{self.server_url}/api/network/register",
                    json=payload,
                    timeout=30,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        neighbors = data.get("neighbors", [])

                        for n in neighbors:
                            if n["id"] != self.my_id:
                                self._add_or_update_neighbor(n)

        except ImportError:
            logger.warning("aiohttp not available for neighbor discovery")
        except Exception as e:
            logger.error(f"Neighbor discovery failed: {e}")

    async def _connect_to_server(self):
        """Connect to relay server via WebSocket for real-time alerts."""
        if not self.server_url:
            return

        try:
            import aiohttp

            ws_url = self.server_url.replace("http", "ws") + "/api/network/ws"

            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    # Authenticate
                    await ws.send_json({
                        "type": "auth",
                        "id": self.my_id,
                        "location": self._blur_location(self.my_location),
                    })

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("type") == "alert":
                                alert = SharedAlert.from_dict(data["alert"])
                                await self._handle_incoming_alert(alert)

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"WebSocket connection error: {e}")

    def _add_or_update_neighbor(self, data: Dict[str, Any]):
        """Add or update a neighbor."""
        neighbor_id = data["id"]

        if neighbor_id in self.neighbors:
            neighbor = self.neighbors[neighbor_id]
            neighbor.last_seen = time.time()
            if data.get("location"):
                neighbor.location = data["location"]
                neighbor.distance_km = self._calculate_distance(data["location"])
        else:
            neighbor = Neighbor(
                id=neighbor_id,
                name=data.get("name", "Unknown"),
                location=data.get("location"),
                distance_km=self._calculate_distance(data.get("location")) if data.get("location") else None,
                trust_level=TrustLevel.PENDING,
            )
            self.neighbors[neighbor_id] = neighbor
            logger.info(f"Discovered new neighbor: {neighbor.name}")

        self._save_data()

    def _calculate_distance(self, location: Dict[str, float]) -> float:
        """Calculate distance to a location in km."""
        lat1, lon1 = self.my_location["lat"], self.my_location["lng"]
        lat2, lon2 = location["lat"], location["lng"]

        R = 6371  # Earth radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _blur_location(self, location: Dict[str, float]) -> Dict[str, float]:
        """Blur location for privacy."""
        if not self.config.anonymous_mode:
            return location

        import random
        # Add random offset within blur radius
        blur_deg = self.config.location_blur_meters / 111000  # Approx meters to degrees
        return {
            "lat": location["lat"] + random.uniform(-blur_deg, blur_deg),
            "lng": location["lng"] + random.uniform(-blur_deg, blur_deg),
        }

    # Neighbor management

    def get_neighbor(self, neighbor_id: str) -> Optional[Neighbor]:
        """Get a neighbor by ID."""
        return self.neighbors.get(neighbor_id)

    def list_neighbors(
        self,
        trust_level: Optional[TrustLevel] = None,
        max_distance_km: Optional[float] = None,
    ) -> List[Neighbor]:
        """List neighbors with optional filters."""
        neighbors = list(self.neighbors.values())

        if trust_level:
            neighbors = [n for n in neighbors if n.trust_level == trust_level]

        if max_distance_km:
            neighbors = [n for n in neighbors if n.distance_km and n.distance_km <= max_distance_km]

        return sorted(neighbors, key=lambda n: n.distance_km or float("inf"))

    def set_trust_level(self, neighbor_id: str, level: TrustLevel):
        """Set trust level for a neighbor."""
        neighbor = self.neighbors.get(neighbor_id)
        if neighbor:
            neighbor.trust_level = level
            self._save_data()
            logger.info(f"Set trust level for {neighbor.name}: {level.value}")

    def block_neighbor(self, neighbor_id: str):
        """Block a neighbor."""
        self.set_trust_level(neighbor_id, TrustLevel.BLOCKED)

    def remove_neighbor(self, neighbor_id: str):
        """Remove a neighbor."""
        if neighbor_id in self.neighbors:
            del self.neighbors[neighbor_id]
            self._save_data()

    # Alert sharing

    async def share_alert(
        self,
        predator_type: str,
        confidence: float,
        location: Dict[str, float],
        heading: Optional[float] = None,
        speed_estimate: Optional[float] = None,
        description: Optional[str] = None,
    ) -> Optional[SharedAlert]:
        """
        Share a predator alert with neighbors.

        Returns:
            SharedAlert if shared, None if not shared
        """
        if not self.config.enabled or not self.config.auto_share_alerts:
            return None

        if confidence < self.config.min_confidence_to_share:
            return None

        if predator_type not in self.config.share_predator_types:
            return None

        import uuid
        alert_id = f"alert-{uuid.uuid4().hex[:12]}"

        alert = SharedAlert(
            id=alert_id,
            source_id=self.my_id,
            predator_type=predator_type,
            confidence=confidence,
            timestamp=time.time(),
            location=self._blur_location(location),
            heading=heading,
            speed_estimate=speed_estimate,
            description=description,
        )

        self.sent_alert_ids.add(alert_id)

        # Send to relay server
        if self.server_url:
            await self._send_alert_to_server(alert)

        # Send directly to trusted neighbors
        await self._send_alert_to_neighbors(alert)

        logger.info(f"Shared alert: {predator_type} at {location}")
        return alert

    async def _send_alert_to_server(self, alert: SharedAlert):
        """Send alert to relay server for distribution."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/api/network/alert",
                    json={
                        "sender_id": self.my_id,
                        "alert": alert.to_dict(),
                        "radius_km": self.config.share_radius_km,
                    },
                    timeout=10,
                ) as response:
                    if response.status == 200:
                        logger.debug("Alert sent to server")

        except Exception as e:
            logger.error(f"Failed to send alert to server: {e}")

    async def _send_alert_to_neighbors(self, alert: SharedAlert):
        """Send alert directly to trusted neighbors."""
        for neighbor in self.neighbors.values():
            if neighbor.trust_level in [TrustLevel.TRUSTED, TrustLevel.VERIFIED]:
                if neighbor.distance_km and neighbor.distance_km <= self.config.share_radius_km:
                    # Would send via direct connection if available
                    neighbor.alerts_sent += 1

        self._save_data()

    async def _handle_incoming_alert(self, alert: SharedAlert):
        """Handle an incoming alert from the network."""
        # Skip if we've seen this alert
        if alert.id in self.received_alerts:
            return

        # Skip if it's our own alert
        if alert.source_id == self.my_id:
            return

        # Skip expired alerts
        if alert.is_expired():
            return

        # Skip if source is blocked
        source = self.neighbors.get(alert.source_id)
        if source and source.trust_level == TrustLevel.BLOCKED:
            return

        # Store alert
        self.received_alerts[alert.id] = alert

        # Update neighbor stats
        if source:
            source.alerts_received += 1
            source.last_seen = time.time()
            self._save_data()

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                result = callback(alert)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.info(f"Received alert: {alert.predator_type} from {alert.source_id}")

        # Relay if appropriate
        if alert.hops < self.MAX_HOPS:
            await self._relay_alert(alert)

    async def _relay_alert(self, alert: SharedAlert):
        """Relay an alert to extend range."""
        if not self.config.enabled:
            return

        relayed_alert = SharedAlert(
            id=alert.id,
            source_id=alert.source_id,
            predator_type=alert.predator_type,
            confidence=alert.confidence,
            timestamp=alert.timestamp,
            location=alert.location,
            heading=alert.heading,
            speed_estimate=alert.speed_estimate,
            description=alert.description,
            expires_at=alert.expires_at,
            hops=alert.hops + 1,
            verified=alert.verified,
        )

        if self.server_url:
            await self._send_alert_to_server(relayed_alert)

    def _cleanup_expired_alerts(self):
        """Remove expired alerts."""
        expired = [
            alert_id for alert_id, alert in self.received_alerts.items()
            if alert.is_expired()
        ]
        for alert_id in expired:
            del self.received_alerts[alert_id]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired alerts")

    def _cleanup_stale_neighbors(self):
        """Remove neighbors not seen in a long time."""
        stale_threshold = time.time() - 86400 * 7  # 7 days
        stale = [
            n_id for n_id, n in self.neighbors.items()
            if n.last_seen < stale_threshold and n.trust_level == TrustLevel.PENDING
        ]
        for n_id in stale:
            del self.neighbors[n_id]

        if stale:
            self._save_data()
            logger.debug(f"Removed {len(stale)} stale neighbors")

    # Alert retrieval

    def get_active_alerts(self) -> List[SharedAlert]:
        """Get all active (non-expired) alerts."""
        return [
            alert for alert in self.received_alerts.values()
            if not alert.is_expired()
        ]

    def get_nearby_alerts(self, radius_km: float = 5.0) -> List[SharedAlert]:
        """Get active alerts within radius."""
        alerts = []
        for alert in self.get_active_alerts():
            distance = self._calculate_distance(alert.location)
            if distance <= radius_km:
                alerts.append(alert)
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            "neighbors_total": len(self.neighbors),
            "neighbors_trusted": sum(
                1 for n in self.neighbors.values()
                if n.trust_level in [TrustLevel.TRUSTED, TrustLevel.VERIFIED]
            ),
            "neighbors_pending": sum(
                1 for n in self.neighbors.values()
                if n.trust_level == TrustLevel.PENDING
            ),
            "active_alerts": len(self.get_active_alerts()),
            "alerts_sent": len(self.sent_alert_ids),
            "alerts_received": sum(n.alerts_received for n in self.neighbors.values()),
        }
