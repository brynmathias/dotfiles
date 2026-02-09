"""
Multi-site support for Garden Sentinel.

Manages multiple garden sites from a single server instance.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from enum import Enum
import asyncio
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SiteLocation:
    """Geographic location of a site."""
    lat: float
    lng: float
    address: Optional[str] = None
    timezone: str = "UTC"


@dataclass
class SiteConfig:
    """Configuration for a site."""
    detection_sensitivity: float = 0.7
    deterrence_enabled: bool = True
    alert_enabled: bool = True
    recording_enabled: bool = True
    quiet_hours_start: Optional[int] = None  # Hour (0-23)
    quiet_hours_end: Optional[int] = None
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class SiteStatus(Enum):
    """Status of a site."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"  # Some devices offline
    MAINTENANCE = "maintenance"


@dataclass
class Site:
    """A garden site with its own devices and configuration."""
    id: str
    name: str
    location: SiteLocation
    config: SiteConfig = field(default_factory=SiteConfig)
    status: SiteStatus = SiteStatus.OFFLINE
    device_ids: Set[str] = field(default_factory=set)
    zone_ids: Set[str] = field(default_factory=set)
    owner_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "location": {
                "lat": self.location.lat,
                "lng": self.location.lng,
                "address": self.location.address,
                "timezone": self.location.timezone,
            },
            "config": {
                "detection_sensitivity": self.config.detection_sensitivity,
                "deterrence_enabled": self.config.deterrence_enabled,
                "alert_enabled": self.config.alert_enabled,
                "recording_enabled": self.config.recording_enabled,
                "quiet_hours_start": self.config.quiet_hours_start,
                "quiet_hours_end": self.config.quiet_hours_end,
                "custom_settings": self.config.custom_settings,
            },
            "status": self.status.value,
            "device_ids": list(self.device_ids),
            "zone_ids": list(self.zone_ids),
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Site":
        location = SiteLocation(
            lat=data["location"]["lat"],
            lng=data["location"]["lng"],
            address=data["location"].get("address"),
            timezone=data["location"].get("timezone", "UTC"),
        )
        config = SiteConfig(
            detection_sensitivity=data["config"].get("detection_sensitivity", 0.7),
            deterrence_enabled=data["config"].get("deterrence_enabled", True),
            alert_enabled=data["config"].get("alert_enabled", True),
            recording_enabled=data["config"].get("recording_enabled", True),
            quiet_hours_start=data["config"].get("quiet_hours_start"),
            quiet_hours_end=data["config"].get("quiet_hours_end"),
            custom_settings=data["config"].get("custom_settings", {}),
        )
        return cls(
            id=data["id"],
            name=data["name"],
            location=location,
            config=config,
            status=SiteStatus(data.get("status", "offline")),
            device_ids=set(data.get("device_ids", [])),
            zone_ids=set(data.get("zone_ids", [])),
            owner_id=data.get("owner_id"),
            created_at=data.get("created_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
        )


@dataclass
class SiteStats:
    """Statistics for a site."""
    site_id: str
    devices_online: int = 0
    devices_total: int = 0
    detections_today: int = 0
    detections_week: int = 0
    alerts_unacknowledged: int = 0
    deterrence_success_rate: float = 0.0
    last_detection: Optional[float] = None
    predator_types_seen: List[str] = field(default_factory=list)


class SiteManager:
    """
    Manages multiple garden sites.

    Features:
    - Site CRUD operations
    - Device-to-site mapping
    - Cross-site aggregation
    - Site-specific configurations
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.sites: Dict[str, Site] = {}
        self.device_site_map: Dict[str, str] = {}  # device_id -> site_id
        self._load_sites()

    def _load_sites(self):
        """Load sites from storage."""
        sites_file = self.storage_path / "sites.json"
        if sites_file.exists():
            with open(sites_file) as f:
                data = json.load(f)
                for site_data in data:
                    site = Site.from_dict(site_data)
                    self.sites[site.id] = site
                    for device_id in site.device_ids:
                        self.device_site_map[device_id] = site.id
            logger.info(f"Loaded {len(self.sites)} sites")

    def _save_sites(self):
        """Save sites to storage."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        sites_file = self.storage_path / "sites.json"
        with open(sites_file, "w") as f:
            json.dump([s.to_dict() for s in self.sites.values()], f, indent=2)

    def create_site(
        self,
        name: str,
        lat: float,
        lng: float,
        owner_id: Optional[str] = None,
        **kwargs,
    ) -> Site:
        """Create a new site."""
        import uuid
        site_id = f"site-{uuid.uuid4().hex[:8]}"

        location = SiteLocation(
            lat=lat,
            lng=lng,
            address=kwargs.get("address"),
            timezone=kwargs.get("timezone", "UTC"),
        )

        config = SiteConfig(
            detection_sensitivity=kwargs.get("detection_sensitivity", 0.7),
            deterrence_enabled=kwargs.get("deterrence_enabled", True),
            alert_enabled=kwargs.get("alert_enabled", True),
            recording_enabled=kwargs.get("recording_enabled", True),
        )

        site = Site(
            id=site_id,
            name=name,
            location=location,
            config=config,
            owner_id=owner_id,
        )

        self.sites[site_id] = site
        self._save_sites()
        logger.info(f"Created site: {name} ({site_id})")

        return site

    def get_site(self, site_id: str) -> Optional[Site]:
        """Get a site by ID."""
        return self.sites.get(site_id)

    def get_site_for_device(self, device_id: str) -> Optional[Site]:
        """Get the site a device belongs to."""
        site_id = self.device_site_map.get(device_id)
        if site_id:
            return self.sites.get(site_id)
        return None

    def list_sites(self, owner_id: Optional[str] = None) -> List[Site]:
        """List all sites, optionally filtered by owner."""
        sites = list(self.sites.values())
        if owner_id:
            sites = [s for s in sites if s.owner_id == owner_id]
        return sites

    def update_site(self, site_id: str, **kwargs) -> Optional[Site]:
        """Update site properties."""
        site = self.sites.get(site_id)
        if not site:
            return None

        if "name" in kwargs:
            site.name = kwargs["name"]
        if "lat" in kwargs:
            site.location.lat = kwargs["lat"]
        if "lng" in kwargs:
            site.location.lng = kwargs["lng"]
        if "address" in kwargs:
            site.location.address = kwargs["address"]
        if "timezone" in kwargs:
            site.location.timezone = kwargs["timezone"]

        # Config updates
        if "detection_sensitivity" in kwargs:
            site.config.detection_sensitivity = kwargs["detection_sensitivity"]
        if "deterrence_enabled" in kwargs:
            site.config.deterrence_enabled = kwargs["deterrence_enabled"]
        if "alert_enabled" in kwargs:
            site.config.alert_enabled = kwargs["alert_enabled"]
        if "recording_enabled" in kwargs:
            site.config.recording_enabled = kwargs["recording_enabled"]
        if "quiet_hours_start" in kwargs:
            site.config.quiet_hours_start = kwargs["quiet_hours_start"]
        if "quiet_hours_end" in kwargs:
            site.config.quiet_hours_end = kwargs["quiet_hours_end"]

        self._save_sites()
        return site

    def delete_site(self, site_id: str) -> bool:
        """Delete a site."""
        site = self.sites.get(site_id)
        if not site:
            return False

        # Remove device mappings
        for device_id in site.device_ids:
            if device_id in self.device_site_map:
                del self.device_site_map[device_id]

        del self.sites[site_id]
        self._save_sites()
        logger.info(f"Deleted site: {site_id}")
        return True

    def assign_device(self, device_id: str, site_id: str) -> bool:
        """Assign a device to a site."""
        site = self.sites.get(site_id)
        if not site:
            return False

        # Remove from old site if assigned
        old_site_id = self.device_site_map.get(device_id)
        if old_site_id and old_site_id in self.sites:
            self.sites[old_site_id].device_ids.discard(device_id)

        site.device_ids.add(device_id)
        self.device_site_map[device_id] = site_id
        self._save_sites()

        logger.info(f"Assigned device {device_id} to site {site_id}")
        return True

    def unassign_device(self, device_id: str) -> bool:
        """Remove a device from its site."""
        site_id = self.device_site_map.get(device_id)
        if not site_id:
            return False

        site = self.sites.get(site_id)
        if site:
            site.device_ids.discard(device_id)

        del self.device_site_map[device_id]
        self._save_sites()

        logger.info(f"Unassigned device {device_id} from site {site_id}")
        return True

    def assign_zone(self, zone_id: str, site_id: str) -> bool:
        """Assign a zone to a site."""
        site = self.sites.get(site_id)
        if not site:
            return False

        site.zone_ids.add(zone_id)
        self._save_sites()
        return True

    def update_site_status(self, site_id: str, device_statuses: Dict[str, bool]):
        """
        Update site status based on device statuses.

        Args:
            site_id: Site ID
            device_statuses: Dict of device_id -> is_online
        """
        site = self.sites.get(site_id)
        if not site:
            return

        online_count = sum(1 for d in site.device_ids if device_statuses.get(d, False))
        total_count = len(site.device_ids)

        if total_count == 0:
            site.status = SiteStatus.OFFLINE
        elif online_count == total_count:
            site.status = SiteStatus.ONLINE
        elif online_count > 0:
            site.status = SiteStatus.DEGRADED
        else:
            site.status = SiteStatus.OFFLINE

        site.last_activity = time.time()

    def get_site_stats(self, site_id: str, db_connection=None) -> SiteStats:
        """Get statistics for a site."""
        site = self.sites.get(site_id)
        if not site:
            return SiteStats(site_id=site_id)

        stats = SiteStats(
            site_id=site_id,
            devices_total=len(site.device_ids),
        )

        # Would query database for actual stats
        # This is a placeholder implementation

        return stats

    def get_aggregate_stats(self, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated statistics across all sites."""
        sites = self.list_sites(owner_id)

        return {
            "total_sites": len(sites),
            "sites_online": sum(1 for s in sites if s.status == SiteStatus.ONLINE),
            "sites_degraded": sum(1 for s in sites if s.status == SiteStatus.DEGRADED),
            "sites_offline": sum(1 for s in sites if s.status == SiteStatus.OFFLINE),
            "total_devices": sum(len(s.device_ids) for s in sites),
            "total_zones": sum(len(s.zone_ids) for s in sites),
        }

    def is_in_quiet_hours(self, site_id: str) -> bool:
        """Check if site is currently in quiet hours."""
        site = self.sites.get(site_id)
        if not site or site.config.quiet_hours_start is None:
            return False

        from datetime import datetime
        import pytz

        try:
            tz = pytz.timezone(site.location.timezone)
            now = datetime.now(tz)
            current_hour = now.hour

            start = site.config.quiet_hours_start
            end = site.config.quiet_hours_end or site.config.quiet_hours_start

            if start <= end:
                return start <= current_hour < end
            else:
                # Overnight quiet hours (e.g., 22:00 - 06:00)
                return current_hour >= start or current_hour < end

        except Exception:
            return False

    def get_site_config_for_device(self, device_id: str) -> Optional[SiteConfig]:
        """Get site configuration for a device."""
        site = self.get_site_for_device(device_id)
        if site:
            return site.config
        return None
