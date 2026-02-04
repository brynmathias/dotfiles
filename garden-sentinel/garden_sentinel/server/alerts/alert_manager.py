"""
Alert management system for Garden Sentinel.
Handles notifications, device commands, and alert storage.
"""

import asyncio
import json
import logging
import smtplib
import threading
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

from garden_sentinel.shared import AlertEvent, CommandType, ServerCommand, ThreatLevel


@dataclass
class NotificationConfig:
    pushover_enabled: bool = False
    pushover_user_key: str = ""
    pushover_api_token: str = ""

    email_enabled: bool = False
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: list[str] = field(default_factory=list)

    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: dict = field(default_factory=dict)

    home_assistant_enabled: bool = False
    home_assistant_url: str = ""
    home_assistant_token: str = ""


@dataclass
class ThreatLevelActions:
    log: bool = True
    notify: bool = False
    alarm: bool = False
    sprayer: bool = False


@dataclass
class AlertConfig:
    cooldown_s: int = 30

    low: ThreatLevelActions = field(default_factory=lambda: ThreatLevelActions(notify=False))
    medium: ThreatLevelActions = field(default_factory=lambda: ThreatLevelActions(notify=True))
    high: ThreatLevelActions = field(default_factory=lambda: ThreatLevelActions(notify=True, alarm=True))
    critical: ThreatLevelActions = field(default_factory=lambda: ThreatLevelActions(notify=True, alarm=True, sprayer=True))

    notifications: NotificationConfig = field(default_factory=NotificationConfig)


class AlertManager:
    """
    Manages alerts, notifications, and device commands.
    """

    def __init__(self, config: AlertConfig):
        self.config = config
        self._command_callback: Optional[Callable[[ServerCommand], None]] = None
        self._alert_history: list[AlertEvent] = []
        self._max_history = 1000

    def handle_alert(self, alert: AlertEvent, frame: Optional[np.ndarray] = None):
        """
        Handle an alert event by executing appropriate actions.
        """
        # Add to history
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]

        # Get actions for threat level
        actions = self._get_actions(alert.max_threat_level)

        # Log
        if actions.log:
            self._log_alert(alert)

        # Execute actions in background
        threading.Thread(
            target=self._execute_actions,
            args=(alert, actions, frame),
            daemon=True,
        ).start()

    def _get_actions(self, threat_level: ThreatLevel) -> ThreatLevelActions:
        """Get actions for a threat level."""
        action_map = {
            ThreatLevel.LOW: self.config.low,
            ThreatLevel.MEDIUM: self.config.medium,
            ThreatLevel.HIGH: self.config.high,
            ThreatLevel.CRITICAL: self.config.critical,
        }
        return action_map.get(threat_level, self.config.medium)

    def _log_alert(self, alert: AlertEvent):
        """Log an alert event."""
        detection_summary = ", ".join(
            f"{d.class_name} ({d.confidence:.2f})" for d in alert.detections
        )
        logger.warning(
            f"ALERT [{alert.max_threat_level.value.upper()}] "
            f"Device: {alert.device_id} | "
            f"Detections: {detection_summary}"
        )

    def _execute_actions(
        self,
        alert: AlertEvent,
        actions: ThreatLevelActions,
        frame: Optional[np.ndarray],
    ):
        """Execute alert actions."""
        actions_taken = []

        try:
            # Send notifications
            if actions.notify:
                self._send_notifications(alert, frame)
                actions_taken.append("notification_sent")

            # Activate alarm
            if actions.alarm and self._command_callback:
                command = ServerCommand(
                    target_device=alert.device_id,
                    command_type=CommandType.ACTIVATE_ALARM,
                    parameters={"duration_s": 10},
                )
                self._command_callback(command)
                actions_taken.append("alarm_activated")

            # Activate sprayer
            if actions.sprayer and self._command_callback:
                command = ServerCommand(
                    target_device=alert.device_id,
                    command_type=CommandType.ACTIVATE_SPRAYER,
                    parameters={"duration_s": 5},
                )
                self._command_callback(command)
                actions_taken.append("sprayer_activated")

            alert.actions_taken = actions_taken

        except Exception as e:
            logger.error(f"Error executing alert actions: {e}")

    def _send_notifications(self, alert: AlertEvent, frame: Optional[np.ndarray]):
        """Send notifications through all configured channels."""
        notif_config = self.config.notifications

        # Prepare message
        detection_list = "\n".join(
            f"  - {d.class_name}: {d.confidence:.1%} confidence"
            for d in alert.detections
        )

        message = (
            f"🚨 PREDATOR ALERT - {alert.max_threat_level.value.upper()}\n\n"
            f"Device: {alert.device_id}\n"
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Detections:\n{detection_list}"
        )

        # Encode frame as JPEG if available
        jpeg_bytes = None
        if frame is not None:
            _, jpeg_data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg_bytes = jpeg_data.tobytes()

        # Send through each channel
        if notif_config.pushover_enabled:
            self._send_pushover(message, jpeg_bytes, alert)

        if notif_config.email_enabled:
            self._send_email(message, jpeg_bytes, alert)

        if notif_config.webhook_enabled:
            self._send_webhook(message, alert)

        if notif_config.home_assistant_enabled:
            self._send_home_assistant(message, alert)

    def _send_pushover(self, message: str, image: Optional[bytes], alert: AlertEvent):
        """Send Pushover notification."""
        try:
            notif = self.config.notifications

            data = {
                "token": notif.pushover_api_token,
                "user": notif.pushover_user_key,
                "message": message,
                "title": f"Garden Sentinel Alert - {alert.max_threat_level.value}",
                "priority": 1 if alert.max_threat_level == ThreatLevel.CRITICAL else 0,
                "sound": "siren" if alert.max_threat_level == ThreatLevel.CRITICAL else "pushover",
            }

            files = {}
            if image:
                files["attachment"] = ("alert.jpg", image, "image/jpeg")

            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data=data,
                files=files if files else None,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info("Pushover notification sent")
            else:
                logger.warning(f"Pushover notification failed: {response.text}")

        except Exception as e:
            logger.error(f"Pushover error: {e}")

    def _send_email(self, message: str, image: Optional[bytes], alert: AlertEvent):
        """Send email notification."""
        try:
            notif = self.config.notifications

            msg = MIMEMultipart()
            msg["Subject"] = f"[Garden Sentinel] {alert.max_threat_level.value.upper()} Alert - {alert.device_id}"
            msg["From"] = notif.email_from
            msg["To"] = ", ".join(notif.email_to)

            # Add text body
            msg.attach(MIMEText(message, "plain"))

            # Add image if available
            if image:
                img_attachment = MIMEImage(image, name="alert.jpg")
                msg.attach(img_attachment)

            # Send email
            with smtplib.SMTP(notif.email_smtp_host, notif.email_smtp_port) as server:
                server.starttls()
                server.login(notif.email_username, notif.email_password)
                server.send_message(msg)

            logger.info(f"Email notification sent to {len(notif.email_to)} recipients")

        except Exception as e:
            logger.error(f"Email error: {e}")

    def _send_webhook(self, message: str, alert: AlertEvent):
        """Send webhook notification."""
        try:
            notif = self.config.notifications

            payload = {
                "event_id": alert.event_id,
                "device_id": alert.device_id,
                "timestamp": alert.timestamp.isoformat(),
                "threat_level": alert.max_threat_level.value,
                "message": message,
                "detections": [d.to_dict() for d in alert.detections],
            }

            response = requests.post(
                notif.webhook_url,
                json=payload,
                headers=notif.webhook_headers,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info("Webhook notification sent")
            else:
                logger.warning(f"Webhook notification failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Webhook error: {e}")

    def _send_home_assistant(self, message: str, alert: AlertEvent):
        """Send Home Assistant notification."""
        try:
            notif = self.config.notifications

            # Call notify service
            response = requests.post(
                f"{notif.home_assistant_url}/api/services/notify/notify",
                json={
                    "message": message,
                    "title": f"Garden Sentinel - {alert.max_threat_level.value}",
                    "data": {
                        "push": {
                            "sound": {
                                "name": "US-EN-Morgan-Freeman-Motion-Detected.wav"
                                if alert.max_threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
                                else "default"
                            }
                        }
                    },
                },
                headers={
                    "Authorization": f"Bearer {notif.home_assistant_token}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )

            # Also fire an event for automations
            requests.post(
                f"{notif.home_assistant_url}/api/events/garden_sentinel_alert",
                json={
                    "device_id": alert.device_id,
                    "threat_level": alert.max_threat_level.value,
                    "detections": [d.class_name for d in alert.detections],
                },
                headers={
                    "Authorization": f"Bearer {notif.home_assistant_token}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )

            logger.info("Home Assistant notification sent")

        except Exception as e:
            logger.error(f"Home Assistant error: {e}")

    def set_command_callback(self, callback: Callable[[ServerCommand], None]):
        """Set the callback for sending commands to devices."""
        self._command_callback = callback

    def get_alert_history(self, limit: int = 100, device_id: Optional[str] = None) -> list[AlertEvent]:
        """Get recent alert history."""
        history = self._alert_history

        if device_id:
            history = [a for a in history if a.device_id == device_id]

        return history[-limit:]

    def send_manual_command(
        self,
        device_id: str,
        command_type: CommandType,
        parameters: Optional[dict] = None,
    ):
        """Send a manual command to a device."""
        if not self._command_callback:
            logger.warning("No command callback set")
            return

        command = ServerCommand(
            target_device=device_id,
            command_type=command_type,
            parameters=parameters or {},
        )

        self._command_callback(command)
        logger.info(f"Manual command sent: {command_type.value} to {device_id}")
