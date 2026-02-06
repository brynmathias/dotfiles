"""
Mobile push notification providers.

Supports:
- Pushover (https://pushover.net) - Paid, very reliable
- Ntfy (https://ntfy.sh) - Free, self-hostable
- Gotify (https://gotify.net) - Free, self-hostable
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import aiohttp

from ...shared.protocol import AlertEvent, ThreatLevel

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOWEST = -2
    LOW = -1
    NORMAL = 0
    HIGH = 1
    EMERGENCY = 2  # Bypasses Do Not Disturb


@dataclass
class NotificationAction:
    """Action button for notification."""
    label: str
    action: str  # URL or action identifier
    clear: bool = False  # Clear notification when tapped


@dataclass
class PushNotification:
    """A push notification to send."""
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    image_url: Optional[str] = None
    click_url: Optional[str] = None
    actions: list[NotificationAction] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # For emergency priority
    retry_interval: int = 60  # seconds
    expire_after: int = 3600  # seconds

    @classmethod
    def from_alert_event(
        cls,
        event: AlertEvent,
        dashboard_url: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> "PushNotification":
        """Create notification from alert event."""
        # Map threat level to priority
        priority_map = {
            ThreatLevel.LOW: NotificationPriority.LOW,
            ThreatLevel.MEDIUM: NotificationPriority.NORMAL,
            ThreatLevel.HIGH: NotificationPriority.HIGH,
            ThreatLevel.CRITICAL: NotificationPriority.EMERGENCY,
        }
        priority = priority_map.get(event.threat_level, NotificationPriority.NORMAL)

        # Build title
        threat_emoji = {
            ThreatLevel.LOW: "⚠️",
            ThreatLevel.MEDIUM: "🟡",
            ThreatLevel.HIGH: "🟠",
            ThreatLevel.CRITICAL: "🔴",
        }
        emoji = threat_emoji.get(event.threat_level, "⚠️")
        title = f"{emoji} {event.predator_type.title()} Detected!"

        # Build message
        message = f"A {event.predator_type} was detected"
        if event.device_id:
            message += f" by camera {event.device_id}"
        if event.confidence:
            message += f" (confidence: {event.confidence:.0%})"

        # Actions
        actions = []
        if dashboard_url:
            actions.append(NotificationAction(
                label="View Dashboard",
                action=dashboard_url,
            ))
            actions.append(NotificationAction(
                label="View Live Feed",
                action=f"{dashboard_url}/live/{event.device_id}",
            ))

        # Tags for filtering
        tags = [
            f"predator:{event.predator_type}",
            f"threat:{event.threat_level.value}",
        ]
        if event.device_id:
            tags.append(f"camera:{event.device_id}")

        return cls(
            title=title,
            message=message,
            priority=priority,
            image_url=image_url,
            click_url=dashboard_url,
            actions=actions,
            tags=tags,
        )


class PushProvider(ABC):
    """Abstract base class for push notification providers."""

    @abstractmethod
    async def send(self, notification: PushNotification) -> bool:
        """Send a notification. Returns True if successful."""
        pass

    @abstractmethod
    async def test(self) -> bool:
        """Test the connection. Returns True if successful."""
        pass


class PushoverProvider(PushProvider):
    """
    Pushover push notification provider.

    Requires:
    - User key (from Pushover dashboard)
    - Application token (create app at pushover.net)
    """

    API_URL = "https://api.pushover.net/1/messages.json"

    def __init__(
        self,
        user_key: str,
        app_token: str,
        device: Optional[str] = None,  # Specific device, or None for all
    ):
        self.user_key = user_key
        self.app_token = app_token
        self.device = device

    async def send(self, notification: PushNotification) -> bool:
        """Send notification via Pushover."""
        # Map priority
        priority_map = {
            NotificationPriority.LOWEST: -2,
            NotificationPriority.LOW: -1,
            NotificationPriority.NORMAL: 0,
            NotificationPriority.HIGH: 1,
            NotificationPriority.EMERGENCY: 2,
        }

        data = {
            "token": self.app_token,
            "user": self.user_key,
            "title": notification.title,
            "message": notification.message,
            "priority": priority_map[notification.priority],
        }

        if self.device:
            data["device"] = self.device

        if notification.click_url:
            data["url"] = notification.click_url
            data["url_title"] = "Open Dashboard"

        if notification.image_url:
            data["attachment_base64"] = notification.image_url  # Or fetch and encode

        # Emergency priority requires retry/expire
        if notification.priority == NotificationPriority.EMERGENCY:
            data["retry"] = notification.retry_interval
            data["expire"] = notification.expire_after

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.API_URL, data=data) as response:
                    if response.status == 200:
                        logger.info(f"Pushover notification sent: {notification.title}")
                        return True
                    else:
                        body = await response.text()
                        logger.error(f"Pushover error {response.status}: {body}")
                        return False
        except Exception as e:
            logger.error(f"Pushover send failed: {e}")
            return False

    async def test(self) -> bool:
        """Test Pushover connection."""
        test_notification = PushNotification(
            title="Garden Sentinel Test",
            message="Push notifications are working!",
            priority=NotificationPriority.LOW,
        )
        return await self.send(test_notification)


class NtfyProvider(PushProvider):
    """
    Ntfy push notification provider.

    Can use ntfy.sh (free, public) or self-hosted instance.

    Features:
    - No account required for public instance
    - Action buttons supported
    - Image attachments supported
    """

    def __init__(
        self,
        topic: str,
        server_url: str = "https://ntfy.sh",
        username: Optional[str] = None,
        password: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        self.topic = topic
        self.server_url = server_url.rstrip("/")
        self.username = username
        self.password = password
        self.access_token = access_token

    async def send(self, notification: PushNotification) -> bool:
        """Send notification via Ntfy."""
        url = f"{self.server_url}/{self.topic}"

        # Map priority (ntfy uses 1-5)
        priority_map = {
            NotificationPriority.LOWEST: 1,
            NotificationPriority.LOW: 2,
            NotificationPriority.NORMAL: 3,
            NotificationPriority.HIGH: 4,
            NotificationPriority.EMERGENCY: 5,
        }

        headers = {
            "Title": notification.title,
            "Priority": str(priority_map[notification.priority]),
        }

        if notification.click_url:
            headers["Click"] = notification.click_url

        if notification.image_url:
            headers["Attach"] = notification.image_url

        if notification.tags:
            # Ntfy uses emoji shortcodes as tags
            tag_emojis = []
            for tag in notification.tags:
                if "critical" in tag.lower():
                    tag_emojis.append("rotating_light")
                elif "high" in tag.lower():
                    tag_emojis.append("warning")
                elif "fox" in tag.lower():
                    tag_emojis.append("fox_face")
                elif "cat" in tag.lower():
                    tag_emojis.append("cat")
                elif "bird" in tag.lower():
                    tag_emojis.append("eagle")
            if tag_emojis:
                headers["Tags"] = ",".join(tag_emojis[:3])  # Max 3 tags

        # Action buttons
        if notification.actions:
            action_strs = []
            for action in notification.actions[:3]:  # Max 3 actions
                action_strs.append(f"view, {action.label}, {action.action}")
            headers["Actions"] = "; ".join(action_strs)

        # Auth
        auth = None
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        elif self.username and self.password:
            auth = aiohttp.BasicAuth(self.username, self.password)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=notification.message,
                    headers=headers,
                    auth=auth,
                ) as response:
                    if response.status == 200:
                        logger.info(f"Ntfy notification sent: {notification.title}")
                        return True
                    else:
                        body = await response.text()
                        logger.error(f"Ntfy error {response.status}: {body}")
                        return False
        except Exception as e:
            logger.error(f"Ntfy send failed: {e}")
            return False

    async def test(self) -> bool:
        """Test Ntfy connection."""
        test_notification = PushNotification(
            title="Garden Sentinel Test",
            message="Push notifications are working!",
            priority=NotificationPriority.LOW,
            tags=["test"],
        )
        return await self.send(test_notification)


class GotifyProvider(PushProvider):
    """
    Gotify push notification provider.

    Self-hosted only. Requires Gotify server and app token.
    """

    def __init__(
        self,
        server_url: str,
        app_token: str,
    ):
        self.server_url = server_url.rstrip("/")
        self.app_token = app_token

    async def send(self, notification: PushNotification) -> bool:
        """Send notification via Gotify."""
        url = f"{self.server_url}/message"

        # Map priority (Gotify uses 0-10)
        priority_map = {
            NotificationPriority.LOWEST: 1,
            NotificationPriority.LOW: 3,
            NotificationPriority.NORMAL: 5,
            NotificationPriority.HIGH: 8,
            NotificationPriority.EMERGENCY: 10,
        }

        # Build extras for click action
        extras = {}
        if notification.click_url:
            extras["client::notification"] = {
                "click": {"url": notification.click_url}
            }

        data = {
            "title": notification.title,
            "message": notification.message,
            "priority": priority_map[notification.priority],
        }

        if extras:
            data["extras"] = extras

        headers = {
            "X-Gotify-Key": self.app_token,
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=data,
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        logger.info(f"Gotify notification sent: {notification.title}")
                        return True
                    else:
                        body = await response.text()
                        logger.error(f"Gotify error {response.status}: {body}")
                        return False
        except Exception as e:
            logger.error(f"Gotify send failed: {e}")
            return False

    async def test(self) -> bool:
        """Test Gotify connection."""
        test_notification = PushNotification(
            title="Garden Sentinel Test",
            message="Push notifications are working!",
            priority=NotificationPriority.LOW,
        )
        return await self.send(test_notification)


class PushNotificationManager:
    """
    Manages multiple push notification providers.

    Sends to all configured providers, with optional filtering
    by priority or tags.
    """

    def __init__(self, dashboard_url: Optional[str] = None):
        self.providers: list[PushProvider] = []
        self.dashboard_url = dashboard_url
        self._min_priority = NotificationPriority.LOW

    def add_provider(self, provider: PushProvider):
        """Add a push notification provider."""
        self.providers.append(provider)
        logger.info(f"Added push provider: {type(provider).__name__}")

    def set_minimum_priority(self, priority: NotificationPriority):
        """Set minimum priority for notifications to be sent."""
        self._min_priority = priority

    async def send_alert(
        self,
        event: AlertEvent,
        image_url: Optional[str] = None,
    ) -> int:
        """
        Send alert to all providers.

        Returns number of successful sends.
        """
        notification = PushNotification.from_alert_event(
            event,
            dashboard_url=self.dashboard_url,
            image_url=image_url,
        )

        return await self.send(notification)

    async def send(self, notification: PushNotification) -> int:
        """
        Send notification to all providers.

        Returns number of successful sends.
        """
        # Check minimum priority
        if notification.priority.value < self._min_priority.value:
            logger.debug(f"Notification below minimum priority: {notification.title}")
            return 0

        if not self.providers:
            logger.warning("No push providers configured")
            return 0

        # Send to all providers concurrently
        tasks = [provider.send(notification) for provider in self.providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(
            1 for r in results
            if r is True
        )

        if success_count < len(self.providers):
            failed = len(self.providers) - success_count
            logger.warning(f"Push notification failed for {failed} provider(s)")

        return success_count

    async def test_all(self) -> dict[str, bool]:
        """Test all providers. Returns dict of provider name -> success."""
        results = {}
        for provider in self.providers:
            name = type(provider).__name__
            try:
                results[name] = await provider.test()
            except Exception as e:
                logger.error(f"Provider test failed for {name}: {e}")
                results[name] = False
        return results


def create_push_manager_from_config(config: dict) -> PushNotificationManager:
    """
    Create PushNotificationManager from configuration dict.

    Expected config format:
    {
        "dashboard_url": "http://192.168.1.100:8000",
        "min_priority": "normal",  # lowest, low, normal, high, emergency
        "pushover": {
            "user_key": "xxx",
            "app_token": "xxx",
            "device": null  # optional
        },
        "ntfy": {
            "topic": "garden-sentinel",
            "server_url": "https://ntfy.sh",  # optional
            "username": null,  # optional
            "password": null,  # optional
            "access_token": null  # optional
        },
        "gotify": {
            "server_url": "http://gotify.local",
            "app_token": "xxx"
        }
    }
    """
    manager = PushNotificationManager(
        dashboard_url=config.get("dashboard_url")
    )

    # Set minimum priority
    priority_map = {
        "lowest": NotificationPriority.LOWEST,
        "low": NotificationPriority.LOW,
        "normal": NotificationPriority.NORMAL,
        "high": NotificationPriority.HIGH,
        "emergency": NotificationPriority.EMERGENCY,
    }
    if "min_priority" in config:
        manager.set_minimum_priority(
            priority_map.get(config["min_priority"], NotificationPriority.LOW)
        )

    # Add Pushover
    if "pushover" in config and config["pushover"]:
        po = config["pushover"]
        if po.get("user_key") and po.get("app_token"):
            manager.add_provider(PushoverProvider(
                user_key=po["user_key"],
                app_token=po["app_token"],
                device=po.get("device"),
            ))

    # Add Ntfy
    if "ntfy" in config and config["ntfy"]:
        ntfy = config["ntfy"]
        if ntfy.get("topic"):
            manager.add_provider(NtfyProvider(
                topic=ntfy["topic"],
                server_url=ntfy.get("server_url", "https://ntfy.sh"),
                username=ntfy.get("username"),
                password=ntfy.get("password"),
                access_token=ntfy.get("access_token"),
            ))

    # Add Gotify
    if "gotify" in config and config["gotify"]:
        gotify = config["gotify"]
        if gotify.get("server_url") and gotify.get("app_token"):
            manager.add_provider(GotifyProvider(
                server_url=gotify["server_url"],
                app_token=gotify["app_token"],
            ))

    return manager
