"""
Version handling for OTA updates.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import re


@dataclass
class Version:
    """Semantic version representation."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: "Version") -> bool:
        return compare_versions(self, other) < 0

    def __le__(self, other: "Version") -> bool:
        return compare_versions(self, other) <= 0

    def __gt__(self, other: "Version") -> bool:
        return compare_versions(self, other) > 0

    def __ge__(self, other: "Version") -> bool:
        return compare_versions(self, other) >= 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return False
        return compare_versions(self, other) == 0

    def tuple(self) -> Tuple[int, int, int]:
        """Return version as tuple for comparison."""
        return (self.major, self.minor, self.patch)

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """
        Parse a version string.

        Supports formats:
        - "1.2.3"
        - "1.2.3-alpha.1"
        - "1.2.3+build.123"
        - "1.2.3-beta.2+build.456"
        """
        # Regex for semantic versioning
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$"
        match = re.match(pattern, version_str.strip())

        if not match:
            raise ValueError(f"Invalid version string: {version_str}")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )


def compare_versions(v1: Version, v2: Version) -> int:
    """
    Compare two versions.

    Returns:
        -1 if v1 < v2
         0 if v1 == v2
         1 if v1 > v2
    """
    # Compare major.minor.patch
    for a, b in zip(v1.tuple(), v2.tuple()):
        if a < b:
            return -1
        if a > b:
            return 1

    # If version numbers equal, compare prerelease
    # No prerelease > prerelease (1.0.0 > 1.0.0-alpha)
    if v1.prerelease is None and v2.prerelease is not None:
        return 1
    if v1.prerelease is not None and v2.prerelease is None:
        return -1
    if v1.prerelease is not None and v2.prerelease is not None:
        # Lexicographic comparison of prerelease
        if v1.prerelease < v2.prerelease:
            return -1
        if v1.prerelease > v2.prerelease:
            return 1

    return 0


@dataclass
class VersionInfo:
    """Complete version information for a release."""
    version: Version
    release_date: str
    changelog: str
    min_required_version: Optional[Version] = None
    download_url: str = ""
    checksum_sha256: str = ""
    size_bytes: int = 0
    signature: str = ""  # GPG signature for verification

    def is_compatible(self, current_version: Version) -> bool:
        """Check if update is compatible with current version."""
        if self.min_required_version is None:
            return True
        return current_version >= self.min_required_version

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "version": str(self.version),
            "release_date": self.release_date,
            "changelog": self.changelog,
            "min_required_version": str(self.min_required_version) if self.min_required_version else None,
            "download_url": self.download_url,
            "checksum_sha256": self.checksum_sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VersionInfo":
        """Create from dictionary."""
        return cls(
            version=Version.parse(data["version"]),
            release_date=data["release_date"],
            changelog=data.get("changelog", ""),
            min_required_version=Version.parse(data["min_required_version"])
                if data.get("min_required_version") else None,
            download_url=data.get("download_url", ""),
            checksum_sha256=data.get("checksum_sha256", ""),
            size_bytes=data.get("size_bytes", 0),
            signature=data.get("signature", ""),
        )
