"""
OTA (Over-The-Air) Update Manager for edge devices.

Handles:
- Checking for updates
- Downloading update packages
- Verifying integrity
- Safe installation with rollback
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pathlib import Path
import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time

from .version import Version, VersionInfo, compare_versions

logger = logging.getLogger(__name__)


class UpdateStatus(Enum):
    """Status of update process."""
    IDLE = "idle"
    CHECKING = "checking"
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    INSTALLING = "installing"
    RESTARTING = "restarting"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class UpdateInfo:
    """Information about an available update."""
    available: bool
    current_version: Version
    latest_version: Optional[Version] = None
    version_info: Optional[VersionInfo] = None
    status: UpdateStatus = UpdateStatus.IDLE
    progress: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "available": self.available,
            "current_version": str(self.current_version),
            "latest_version": str(self.latest_version) if self.latest_version else None,
            "status": self.status.value,
            "progress": self.progress,
            "error": self.error,
            "changelog": self.version_info.changelog if self.version_info else None,
            "size_bytes": self.version_info.size_bytes if self.version_info else None,
        }


@dataclass
class UpdateResult:
    """Result of an update operation."""
    success: bool
    previous_version: Version
    new_version: Optional[Version] = None
    message: str = ""
    requires_restart: bool = False


class OTAUpdateManager:
    """
    Manages OTA updates for Garden Sentinel edge devices.

    Features:
    - Automatic update checking
    - Secure download with checksum verification
    - Atomic installation with rollback
    - Scheduled update windows
    - Update history tracking
    """

    UPDATE_CHECK_INTERVAL = 3600  # 1 hour
    DOWNLOAD_CHUNK_SIZE = 8192  # 8KB
    MAX_RETRY_ATTEMPTS = 3

    def __init__(
        self,
        device_id: str,
        current_version: str,
        update_server_url: str,
        install_dir: Path,
        backup_dir: Optional[Path] = None,
    ):
        self.device_id = device_id
        self.current_version = Version.parse(current_version)
        self.update_server_url = update_server_url.rstrip("/")
        self.install_dir = install_dir
        self.backup_dir = backup_dir or install_dir.parent / "backup"

        self.status = UpdateStatus.IDLE
        self.progress = 0.0
        self.error: Optional[str] = None
        self.latest_info: Optional[VersionInfo] = None

        self._update_task: Optional[asyncio.Task] = None
        self._check_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[UpdateInfo], None]] = []
        self._auto_check_enabled = False

        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def add_callback(self, callback: Callable[[UpdateInfo], None]):
        """Add a callback for update status changes."""
        self._callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all callbacks of status change."""
        info = self.get_update_info()
        for callback in self._callbacks:
            try:
                callback(info)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_update_info(self) -> UpdateInfo:
        """Get current update information."""
        return UpdateInfo(
            available=self.latest_info is not None and self.latest_info.version > self.current_version,
            current_version=self.current_version,
            latest_version=self.latest_info.version if self.latest_info else None,
            version_info=self.latest_info,
            status=self.status,
            progress=self.progress,
            error=self.error,
        )

    async def check_for_updates(self) -> UpdateInfo:
        """
        Check the update server for available updates.

        Returns:
            UpdateInfo with current status
        """
        self.status = UpdateStatus.CHECKING
        self.error = None
        self._notify_callbacks()

        try:
            import aiohttp

            url = f"{self.update_server_url}/api/updates/latest"
            params = {
                "device_id": self.device_id,
                "current_version": str(self.current_version),
                "platform": self._get_platform(),
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status != 200:
                        raise Exception(f"Server returned {response.status}")

                    data = await response.json()

            if data.get("update_available"):
                self.latest_info = VersionInfo.from_dict(data["version_info"])
                self.status = UpdateStatus.AVAILABLE
            else:
                self.latest_info = None
                self.status = UpdateStatus.IDLE

            logger.info(f"Update check complete. Available: {self.latest_info is not None}")

        except ImportError:
            # Fallback without aiohttp
            self.status = UpdateStatus.IDLE
            logger.warning("aiohttp not available, skipping update check")

        except Exception as e:
            self.error = str(e)
            self.status = UpdateStatus.FAILED
            logger.error(f"Update check failed: {e}")

        self._notify_callbacks()
        return self.get_update_info()

    async def download_update(self) -> bool:
        """
        Download the update package.

        Returns:
            True if download successful
        """
        if not self.latest_info:
            self.error = "No update available"
            return False

        self.status = UpdateStatus.DOWNLOADING
        self.progress = 0.0
        self._notify_callbacks()

        try:
            import aiohttp

            download_path = self._get_download_path()

            async with aiohttp.ClientSession() as session:
                async with session.get(self.latest_info.download_url, timeout=600) as response:
                    if response.status != 200:
                        raise Exception(f"Download failed: {response.status}")

                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    with open(download_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(self.DOWNLOAD_CHUNK_SIZE):
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total_size > 0:
                                self.progress = downloaded / total_size
                                self._notify_callbacks()

            logger.info(f"Download complete: {download_path}")
            self.progress = 1.0
            return True

        except ImportError:
            self.error = "aiohttp not available"
            self.status = UpdateStatus.FAILED
            return False

        except Exception as e:
            self.error = str(e)
            self.status = UpdateStatus.FAILED
            logger.error(f"Download failed: {e}")
            return False

    def verify_update(self) -> bool:
        """
        Verify the downloaded update package.

        Returns:
            True if verification successful
        """
        if not self.latest_info:
            return False

        self.status = UpdateStatus.VERIFYING
        self._notify_callbacks()

        try:
            download_path = self._get_download_path()

            if not download_path.exists():
                raise Exception("Update file not found")

            # Verify checksum
            sha256 = hashlib.sha256()
            with open(download_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)

            calculated_hash = sha256.hexdigest()
            expected_hash = self.latest_info.checksum_sha256

            if calculated_hash != expected_hash:
                raise Exception(
                    f"Checksum mismatch: expected {expected_hash}, got {calculated_hash}"
                )

            logger.info("Update verification successful")
            return True

        except Exception as e:
            self.error = str(e)
            self.status = UpdateStatus.FAILED
            logger.error(f"Verification failed: {e}")
            return False

    async def install_update(self, restart: bool = True) -> UpdateResult:
        """
        Install the downloaded update.

        Args:
            restart: Whether to restart the service after installation

        Returns:
            UpdateResult with status
        """
        if not self.latest_info:
            return UpdateResult(
                success=False,
                previous_version=self.current_version,
                message="No update available",
            )

        self.status = UpdateStatus.INSTALLING
        self._notify_callbacks()

        previous_version = self.current_version
        download_path = self._get_download_path()

        try:
            # Create backup
            backup_path = self._create_backup()
            logger.info(f"Created backup at {backup_path}")

            # Extract and install
            self._extract_and_install(download_path)

            # Update version file
            self._update_version_file(self.latest_info.version)

            # Verify installation
            if not self._verify_installation():
                raise Exception("Installation verification failed")

            self.current_version = self.latest_info.version
            self.status = UpdateStatus.COMPLETED
            self._notify_callbacks()

            logger.info(f"Update installed: {previous_version} -> {self.current_version}")

            # Restart if requested
            if restart:
                self.status = UpdateStatus.RESTARTING
                self._notify_callbacks()
                await self._restart_service()

            return UpdateResult(
                success=True,
                previous_version=previous_version,
                new_version=self.current_version,
                message="Update installed successfully",
                requires_restart=restart,
            )

        except Exception as e:
            self.error = str(e)
            logger.error(f"Installation failed: {e}")

            # Attempt rollback
            try:
                self._rollback()
                self.status = UpdateStatus.ROLLED_BACK
                logger.info("Rollback successful")
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
                self.status = UpdateStatus.FAILED

            self._notify_callbacks()

            return UpdateResult(
                success=False,
                previous_version=previous_version,
                message=f"Installation failed: {e}",
            )

    def _get_download_path(self) -> Path:
        """Get path for downloaded update file."""
        return self.backup_dir / f"update-{self.latest_info.version}.tar.gz"

    def _get_platform(self) -> str:
        """Get platform identifier."""
        import platform
        arch = platform.machine()
        system = platform.system().lower()
        return f"{system}-{arch}"

    def _create_backup(self) -> Path:
        """Create backup of current installation."""
        timestamp = int(time.time())
        backup_path = self.backup_dir / f"backup-{self.current_version}-{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)

        # Copy current installation
        for item in self.install_dir.iterdir():
            if item.name.startswith("."):
                continue
            dest = backup_path / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        # Save backup info
        info = {
            "version": str(self.current_version),
            "timestamp": timestamp,
            "path": str(self.install_dir),
        }
        with open(backup_path / "backup_info.json", "w") as f:
            json.dump(info, f)

        return backup_path

    def _extract_and_install(self, archive_path: Path):
        """Extract update archive and install files."""
        import tarfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Extract archive
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(tmp_path)

            # Find extracted content
            extracted_items = list(tmp_path.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                source_dir = extracted_items[0]
            else:
                source_dir = tmp_path

            # Install files
            for item in source_dir.iterdir():
                dest = self.install_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()

                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

    def _update_version_file(self, version: Version):
        """Update the version file."""
        version_file = self.install_dir / "VERSION"
        with open(version_file, "w") as f:
            f.write(str(version))

    def _verify_installation(self) -> bool:
        """Verify the installation is functional."""
        # Basic check: version file exists and is readable
        version_file = self.install_dir / "VERSION"
        if not version_file.exists():
            return False

        # Check main entry point exists
        main_file = self.install_dir / "garden_sentinel" / "edge" / "main.py"
        if not main_file.exists():
            main_file = self.install_dir / "main.py"

        return main_file.exists()

    def _rollback(self):
        """Rollback to previous version from backup."""
        # Find most recent backup
        backups = sorted(
            self.backup_dir.glob("backup-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not backups:
            raise Exception("No backup available for rollback")

        backup_path = backups[0]
        logger.info(f"Rolling back from {backup_path}")

        # Restore from backup
        for item in backup_path.iterdir():
            if item.name == "backup_info.json":
                continue

            dest = self.install_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()

            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        # Restore version
        with open(backup_path / "backup_info.json") as f:
            info = json.load(f)
        self.current_version = Version.parse(info["version"])

    async def _restart_service(self):
        """Restart the Garden Sentinel service."""
        logger.info("Restarting service...")

        # Try systemd first
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", "garden-sentinel-edge"],
                check=True,
                capture_output=True,
            )
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Fallback: exit and let supervisor restart
        logger.info("Exiting for restart...")
        await asyncio.sleep(1)
        os._exit(0)

    async def start_auto_check(self, interval: int = UPDATE_CHECK_INTERVAL):
        """Start automatic update checking."""
        if self._auto_check_enabled:
            return

        self._auto_check_enabled = True

        async def check_loop():
            while self._auto_check_enabled:
                try:
                    await self.check_for_updates()
                except Exception as e:
                    logger.error(f"Auto-check error: {e}")
                await asyncio.sleep(interval)

        self._check_task = asyncio.create_task(check_loop())
        logger.info(f"Started auto-check every {interval}s")

    def stop_auto_check(self):
        """Stop automatic update checking."""
        self._auto_check_enabled = False
        if self._check_task:
            self._check_task.cancel()
            self._check_task = None

    def cleanup_old_backups(self, keep_count: int = 3):
        """Remove old backups, keeping most recent."""
        backups = sorted(
            self.backup_dir.glob("backup-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for backup in backups[keep_count:]:
            try:
                shutil.rmtree(backup)
                logger.info(f"Removed old backup: {backup}")
            except Exception as e:
                logger.error(f"Failed to remove backup {backup}: {e}")

    def get_update_history(self) -> List[Dict[str, Any]]:
        """Get history of installed updates from backups."""
        history = []

        for backup in self.backup_dir.glob("backup-*"):
            info_file = backup / "backup_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                    history.append(info)

        return sorted(history, key=lambda x: x["timestamp"], reverse=True)
