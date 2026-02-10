"""
Individual predator tracking and identification.

Maintains profiles of individual animals for behavioral analysis
and targeted deterrence strategies.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import json
import logging
import time
from pathlib import Path

from .feature_extractor import FeatureExtractor, FeatureVector

logger = logging.getLogger(__name__)


@dataclass
class Sighting:
    """A single sighting of an individual."""
    id: str
    timestamp: float
    device_id: str
    site_id: Optional[str]
    confidence: float
    feature_vector: Optional[FeatureVector]
    frame_path: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    position: Optional[Dict[str, float]] = None  # lat, lng if available
    behavior: Optional[str] = None  # detected behavior
    deterrence_applied: Optional[str] = None
    deterrence_effective: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "site_id": self.site_id,
            "confidence": self.confidence,
            "feature_vector": self.feature_vector.to_dict() if self.feature_vector else None,
            "frame_path": self.frame_path,
            "bbox": self.bbox,
            "position": self.position,
            "behavior": self.behavior,
            "deterrence_applied": self.deterrence_applied,
            "deterrence_effective": self.deterrence_effective,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sighting":
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            device_id=data["device_id"],
            site_id=data.get("site_id"),
            confidence=data["confidence"],
            feature_vector=FeatureVector.from_dict(data["feature_vector"]) if data.get("feature_vector") else None,
            frame_path=data.get("frame_path"),
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            position=data.get("position"),
            behavior=data.get("behavior"),
            deterrence_applied=data.get("deterrence_applied"),
            deterrence_effective=data.get("deterrence_effective"),
        )


class ThreatLevel(Enum):
    """Threat level assessment for an individual."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PredatorProfile:
    """Profile of an identified individual predator."""
    id: str
    predator_type: str
    name: Optional[str] = None  # User-assigned name
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    sighting_count: int = 0
    threat_level: ThreatLevel = ThreatLevel.MEDIUM

    # Feature embeddings (averaged from sightings)
    primary_features: Optional[FeatureVector] = None
    feature_samples: List[FeatureVector] = field(default_factory=list)

    # Behavioral data
    typical_times: List[int] = field(default_factory=list)  # Hours of day
    typical_entry_points: List[str] = field(default_factory=list)  # Zone IDs
    sites_visited: List[str] = field(default_factory=list)

    # Deterrence effectiveness
    deterrence_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # e.g., {"spray": {"success": 5, "failure": 2}, "sound": {...}}

    # Notes
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "predator_type": self.predator_type,
            "name": self.name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "sighting_count": self.sighting_count,
            "threat_level": self.threat_level.value,
            "primary_features": self.primary_features.to_dict() if self.primary_features else None,
            "feature_samples": [f.to_dict() for f in self.feature_samples[-5:]],  # Keep last 5
            "typical_times": self.typical_times,
            "typical_entry_points": self.typical_entry_points,
            "sites_visited": self.sites_visited,
            "deterrence_stats": self.deterrence_stats,
            "notes": self.notes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredatorProfile":
        return cls(
            id=data["id"],
            predator_type=data["predator_type"],
            name=data.get("name"),
            first_seen=data.get("first_seen", time.time()),
            last_seen=data.get("last_seen", time.time()),
            sighting_count=data.get("sighting_count", 0),
            threat_level=ThreatLevel(data.get("threat_level", "medium")),
            primary_features=FeatureVector.from_dict(data["primary_features"]) if data.get("primary_features") else None,
            feature_samples=[FeatureVector.from_dict(f) for f in data.get("feature_samples", [])],
            typical_times=data.get("typical_times", []),
            typical_entry_points=data.get("typical_entry_points", []),
            sites_visited=data.get("sites_visited", []),
            deterrence_stats=data.get("deterrence_stats", {}),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
        )

    def update_features(self, new_features: FeatureVector):
        """Update feature embeddings with a new sample."""
        self.feature_samples.append(new_features)

        # Keep only recent samples (sliding window)
        max_samples = 20
        if len(self.feature_samples) > max_samples:
            self.feature_samples = self.feature_samples[-max_samples:]

        # Update primary features (weighted average)
        if len(self.feature_samples) >= 3:
            # Average the vectors
            dim = len(self.feature_samples[0].vector)
            avg_vector = [0.0] * dim

            for sample in self.feature_samples:
                for i, v in enumerate(sample.vector):
                    avg_vector[i] += v

            avg_vector = [v / len(self.feature_samples) for v in avg_vector]

            self.primary_features = FeatureVector(
                vector=avg_vector,
                confidence=min(f.confidence for f in self.feature_samples),
            ).normalize()

    def get_best_deterrence(self) -> Optional[str]:
        """Get the most effective deterrence method for this individual."""
        best_method = None
        best_rate = 0.0

        for method, stats in self.deterrence_stats.items():
            total = stats.get("success", 0) + stats.get("failure", 0)
            if total >= 3:  # Need minimum data
                rate = stats.get("success", 0) / total
                if rate > best_rate:
                    best_rate = rate
                    best_method = method

        return best_method

    def record_deterrence(self, method: str, effective: bool):
        """Record a deterrence attempt result."""
        if method not in self.deterrence_stats:
            self.deterrence_stats[method] = {"success": 0, "failure": 0}

        if effective:
            self.deterrence_stats[method]["success"] += 1
        else:
            self.deterrence_stats[method]["failure"] += 1

        # Update threat level based on deterrence effectiveness
        self._update_threat_level()

    def _update_threat_level(self):
        """Update threat level based on behavior patterns."""
        # More sightings = higher threat
        if self.sighting_count > 20:
            base_threat = ThreatLevel.HIGH
        elif self.sighting_count > 10:
            base_threat = ThreatLevel.MEDIUM
        else:
            base_threat = ThreatLevel.LOW

        # Check deterrence resistance
        total_deterrence = sum(
            s.get("success", 0) + s.get("failure", 0)
            for s in self.deterrence_stats.values()
        )
        if total_deterrence > 0:
            total_failures = sum(s.get("failure", 0) for s in self.deterrence_stats.values())
            failure_rate = total_failures / total_deterrence

            if failure_rate > 0.7:  # Resistant to deterrence
                self.threat_level = ThreatLevel.CRITICAL
                return
            elif failure_rate > 0.5:
                if base_threat == ThreatLevel.HIGH:
                    self.threat_level = ThreatLevel.CRITICAL
                else:
                    self.threat_level = ThreatLevel.HIGH
                return

        self.threat_level = base_threat


@dataclass
class MatchResult:
    """Result of matching a sighting to known individuals."""
    matched: bool
    profile: Optional[PredatorProfile] = None
    confidence: float = 0.0
    is_new: bool = False
    alternatives: List[Tuple[PredatorProfile, float]] = field(default_factory=list)


class IndividualTracker:
    """
    Tracks and identifies individual predators.

    Features:
    - Re-identification across sightings
    - Behavioral pattern analysis
    - Deterrence effectiveness tracking per individual
    - Threat level assessment
    """

    MATCH_THRESHOLD = 0.75  # Minimum similarity to consider a match
    NEW_INDIVIDUAL_THRESHOLD = 0.6  # Below this, likely a new individual
    MAX_ALTERNATIVES = 3

    def __init__(
        self,
        storage_path: Path,
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        self.storage_path = storage_path
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.profiles: Dict[str, PredatorProfile] = {}
        self.sightings: Dict[str, List[Sighting]] = {}  # profile_id -> sightings

        self._load_data()

    def _load_data(self):
        """Load profiles from storage."""
        profiles_file = self.storage_path / "predator_profiles.json"
        if profiles_file.exists():
            with open(profiles_file) as f:
                data = json.load(f)
                for profile_data in data:
                    profile = PredatorProfile.from_dict(profile_data)
                    self.profiles[profile.id] = profile
            logger.info(f"Loaded {len(self.profiles)} predator profiles")

    def _save_data(self):
        """Save profiles to storage."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        profiles_file = self.storage_path / "predator_profiles.json"
        with open(profiles_file, "w") as f:
            json.dump([p.to_dict() for p in self.profiles.values()], f, indent=2)

    def identify(
        self,
        image_data: bytes,
        predator_type: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        device_id: str = "",
        site_id: Optional[str] = None,
    ) -> MatchResult:
        """
        Identify an individual from an image.

        Args:
            image_data: Raw image bytes
            predator_type: Type of predator (fox, hawk, etc.)
            bbox: Bounding box of the animal in the image
            device_id: ID of the camera that captured the image
            site_id: ID of the site

        Returns:
            MatchResult with identified profile or new profile
        """
        # Extract features
        features = self.feature_extractor.extract(image_data, bbox, predator_type)
        if not features or features.confidence < 0.3:
            return MatchResult(matched=False, confidence=0.0)

        # Find matching profiles
        candidates = self._find_candidates(predator_type, features)

        if not candidates:
            # No candidates, create new profile
            profile = self._create_profile(predator_type, features, device_id, site_id)
            return MatchResult(
                matched=True,
                profile=profile,
                confidence=features.confidence,
                is_new=True,
            )

        best_match, best_score = candidates[0]

        if best_score >= self.MATCH_THRESHOLD:
            # Good match found
            self._update_profile(best_match, features, device_id, site_id)
            return MatchResult(
                matched=True,
                profile=best_match,
                confidence=best_score,
                is_new=False,
                alternatives=candidates[1:self.MAX_ALTERNATIVES + 1],
            )
        elif best_score < self.NEW_INDIVIDUAL_THRESHOLD:
            # Likely a new individual
            profile = self._create_profile(predator_type, features, device_id, site_id)
            return MatchResult(
                matched=True,
                profile=profile,
                confidence=features.confidence,
                is_new=True,
                alternatives=candidates[:self.MAX_ALTERNATIVES],
            )
        else:
            # Uncertain - return best match with low confidence
            return MatchResult(
                matched=True,
                profile=best_match,
                confidence=best_score,
                is_new=False,
                alternatives=candidates[1:self.MAX_ALTERNATIVES + 1],
            )

    def _find_candidates(
        self,
        predator_type: str,
        features: FeatureVector,
    ) -> List[Tuple[PredatorProfile, float]]:
        """Find candidate profiles matching the predator type."""
        candidates = []

        for profile in self.profiles.values():
            if profile.predator_type != predator_type:
                continue
            if not profile.primary_features:
                continue

            similarity = self.feature_extractor.compute_similarity(
                features, profile.primary_features
            )
            candidates.append((profile, similarity))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _create_profile(
        self,
        predator_type: str,
        features: FeatureVector,
        device_id: str,
        site_id: Optional[str],
    ) -> PredatorProfile:
        """Create a new predator profile."""
        import uuid

        profile_id = f"pred-{uuid.uuid4().hex[:8]}"

        profile = PredatorProfile(
            id=profile_id,
            predator_type=predator_type,
            primary_features=features.normalize(),
        )
        profile.feature_samples.append(features)
        profile.sighting_count = 1

        if site_id:
            profile.sites_visited.append(site_id)

        # Record time of day
        from datetime import datetime
        hour = datetime.fromtimestamp(features.timestamp or time.time()).hour
        profile.typical_times.append(hour)

        self.profiles[profile_id] = profile
        self._save_data()

        logger.info(f"Created new predator profile: {profile_id} ({predator_type})")
        return profile

    def _update_profile(
        self,
        profile: PredatorProfile,
        features: FeatureVector,
        device_id: str,
        site_id: Optional[str],
    ):
        """Update an existing profile with new sighting."""
        profile.last_seen = time.time()
        profile.sighting_count += 1
        profile.update_features(features)

        if site_id and site_id not in profile.sites_visited:
            profile.sites_visited.append(site_id)

        # Record time of day
        from datetime import datetime
        hour = datetime.fromtimestamp(features.timestamp or time.time()).hour
        if hour not in profile.typical_times:
            profile.typical_times.append(hour)
            # Keep only most common times
            if len(profile.typical_times) > 24:
                # Count occurrences and keep top 12
                from collections import Counter
                counter = Counter(profile.typical_times)
                profile.typical_times = [h for h, _ in counter.most_common(12)]

        self._save_data()
        logger.debug(f"Updated profile {profile.id}, sightings: {profile.sighting_count}")

    def record_deterrence_result(
        self,
        profile_id: str,
        method: str,
        effective: bool,
    ):
        """Record deterrence effectiveness for a profile."""
        profile = self.profiles.get(profile_id)
        if profile:
            profile.record_deterrence(method, effective)
            self._save_data()

    def get_profile(self, profile_id: str) -> Optional[PredatorProfile]:
        """Get a profile by ID."""
        return self.profiles.get(profile_id)

    def list_profiles(
        self,
        predator_type: Optional[str] = None,
        threat_level: Optional[ThreatLevel] = None,
        site_id: Optional[str] = None,
    ) -> List[PredatorProfile]:
        """List profiles with optional filters."""
        profiles = list(self.profiles.values())

        if predator_type:
            profiles = [p for p in profiles if p.predator_type == predator_type]
        if threat_level:
            profiles = [p for p in profiles if p.threat_level == threat_level]
        if site_id:
            profiles = [p for p in profiles if site_id in p.sites_visited]

        return sorted(profiles, key=lambda p: p.last_seen, reverse=True)

    def get_high_threat_individuals(self) -> List[PredatorProfile]:
        """Get individuals with high or critical threat level."""
        return [
            p for p in self.profiles.values()
            if p.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        ]

    def merge_profiles(self, profile_id_1: str, profile_id_2: str) -> Optional[PredatorProfile]:
        """Merge two profiles that are the same individual."""
        p1 = self.profiles.get(profile_id_1)
        p2 = self.profiles.get(profile_id_2)

        if not p1 or not p2:
            return None
        if p1.predator_type != p2.predator_type:
            return None

        # Merge into p1
        p1.first_seen = min(p1.first_seen, p2.first_seen)
        p1.last_seen = max(p1.last_seen, p2.last_seen)
        p1.sighting_count += p2.sighting_count
        p1.feature_samples.extend(p2.feature_samples)
        p1.typical_times.extend(p2.typical_times)
        p1.sites_visited = list(set(p1.sites_visited + p2.sites_visited))
        p1.typical_entry_points = list(set(p1.typical_entry_points + p2.typical_entry_points))

        # Merge deterrence stats
        for method, stats in p2.deterrence_stats.items():
            if method not in p1.deterrence_stats:
                p1.deterrence_stats[method] = stats
            else:
                p1.deterrence_stats[method]["success"] += stats.get("success", 0)
                p1.deterrence_stats[method]["failure"] += stats.get("failure", 0)

        # Update primary features
        if p1.feature_samples:
            p1.update_features(p1.feature_samples[-1])

        # Delete p2
        del self.profiles[profile_id_2]
        self._save_data()

        logger.info(f"Merged profiles {profile_id_2} into {profile_id_1}")
        return p1

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile."""
        if profile_id in self.profiles:
            del self.profiles[profile_id]
            self._save_data()
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        profiles = list(self.profiles.values())

        by_type = {}
        for p in profiles:
            by_type[p.predator_type] = by_type.get(p.predator_type, 0) + 1

        by_threat = {}
        for p in profiles:
            level = p.threat_level.value
            by_threat[level] = by_threat.get(level, 0) + 1

        return {
            "total_profiles": len(profiles),
            "by_type": by_type,
            "by_threat_level": by_threat,
            "total_sightings": sum(p.sighting_count for p in profiles),
        }
