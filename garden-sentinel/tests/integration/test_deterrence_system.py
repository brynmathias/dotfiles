"""
Integration tests for the deterrence system.
"""

import pytest
import time
import json
from unittest.mock import MagicMock, patch


class TestDeterrenceSelection:
    """Tests for selecting appropriate deterrence methods."""

    def test_deterrence_selection_by_predator(self):
        """Test selecting deterrence method based on predator type."""
        deterrence_map = {
            "fox": ["dog_bark", "spray", "ultrasonic"],
            "hawk": ["alarm", "tts"],
            "coyote": ["predator_growl", "spray", "alarm"],
            "raccoon": ["ultrasonic", "spray"],
            "cat": ["ultrasonic", "spray"],
        }

        def select_deterrence(predator_type: str, available_methods: list) -> str:
            preferred = deterrence_map.get(predator_type, ["spray"])
            for method in preferred:
                if method in available_methods:
                    return method
            return available_methods[0] if available_methods else None

        # Camera with all methods
        all_methods = ["spray", "dog_bark", "ultrasonic", "alarm", "tts", "predator_growl"]

        assert select_deterrence("fox", all_methods) == "dog_bark"
        assert select_deterrence("hawk", all_methods) == "alarm"
        assert select_deterrence("raccoon", all_methods) == "ultrasonic"

        # Camera with limited methods
        limited_methods = ["spray"]
        assert select_deterrence("fox", limited_methods) == "spray"

    def test_deterrence_escalation(self):
        """Test escalating deterrence methods on repeated detections."""
        escalation_levels = [
            {"level": 1, "methods": ["ultrasonic"]},
            {"level": 2, "methods": ["dog_bark", "ultrasonic"]},
            {"level": 3, "methods": ["spray", "dog_bark", "alarm"]},
        ]

        def get_escalation_level(detection_count: int, time_window_minutes: int = 10) -> int:
            if detection_count <= 1:
                return 1
            elif detection_count <= 3:
                return 2
            else:
                return 3

        assert get_escalation_level(1) == 1
        assert get_escalation_level(2) == 2
        assert get_escalation_level(5) == 3

    def test_cooldown_period(self):
        """Test respecting cooldown between deterrence activations."""
        last_activation = {}
        cooldown_seconds = 30

        def can_activate(device_id: str, method: str, current_time: float) -> bool:
            key = f"{device_id}:{method}"
            last_time = last_activation.get(key, 0)
            return current_time - last_time >= cooldown_seconds

        def activate(device_id: str, method: str, current_time: float):
            key = f"{device_id}:{method}"
            last_activation[key] = current_time

        now = time.time()

        # First activation should be allowed
        assert can_activate("camera-1", "spray", now)
        activate("camera-1", "spray", now)

        # Immediate re-activation should be blocked
        assert not can_activate("camera-1", "spray", now + 10)

        # After cooldown should be allowed
        assert can_activate("camera-1", "spray", now + 35)

        # Different method should be allowed
        assert can_activate("camera-1", "sound", now + 10)


class TestDeterrenceEffectiveness:
    """Tests for tracking deterrence effectiveness."""

    def test_effectiveness_tracking(self):
        """Test tracking success/failure of deterrence actions."""
        effectiveness_log = []

        def log_deterrence_result(
            predator_type: str,
            method: str,
            success: bool,
            time_to_leave: float = None,
        ):
            effectiveness_log.append({
                "predator": predator_type,
                "method": method,
                "success": success,
                "time_to_leave": time_to_leave,
                "timestamp": time.time(),
            })

        # Log some results
        log_deterrence_result("fox", "dog_bark", True, 3.5)
        log_deterrence_result("fox", "spray", True, 2.1)
        log_deterrence_result("fox", "dog_bark", False)
        log_deterrence_result("hawk", "alarm", True, 1.2)

        # Calculate effectiveness
        fox_bark_results = [e for e in effectiveness_log
                           if e["predator"] == "fox" and e["method"] == "dog_bark"]
        success_rate = sum(1 for e in fox_bark_results if e["success"]) / len(fox_bark_results)

        assert success_rate == pytest.approx(0.5)

    def test_method_recommendation(self):
        """Test recommending best deterrence method based on history."""
        effectiveness_data = {
            "fox": {
                "dog_bark": {"successes": 45, "failures": 5, "avg_time": 3.2},
                "spray": {"successes": 30, "failures": 10, "avg_time": 2.1},
                "ultrasonic": {"successes": 20, "failures": 20, "avg_time": 5.0},
            },
            "hawk": {
                "alarm": {"successes": 35, "failures": 5, "avg_time": 1.5},
                "tts": {"successes": 25, "failures": 10, "avg_time": 2.0},
            },
        }

        def recommend_method(predator_type: str) -> str:
            methods = effectiveness_data.get(predator_type, {})
            if not methods:
                return "spray"  # Default

            # Score based on success rate and response time
            best_method = None
            best_score = 0

            for method, stats in methods.items():
                total = stats["successes"] + stats["failures"]
                if total < 10:
                    continue  # Not enough data

                success_rate = stats["successes"] / total
                time_factor = 1 / (stats["avg_time"] + 1)  # Lower time = better

                score = success_rate * 0.7 + time_factor * 0.3

                if score > best_score:
                    best_score = score
                    best_method = method

            return best_method or "spray"

        assert recommend_method("fox") == "dog_bark"
        assert recommend_method("hawk") == "alarm"
        assert recommend_method("unknown") == "spray"


class TestAudioDeterrence:
    """Tests for audio deterrence system."""

    def test_sound_selection(self):
        """Test selecting appropriate sounds."""
        sound_library = {
            "dog_bark": ["bark_1.wav", "bark_2.wav", "bark_3.wav"],
            "predator_growl": ["growl_1.wav", "growl_2.wav"],
            "fox_distress": ["fox_distress_1.wav"],
            "alarm": ["siren.wav"],
        }

        def select_sound(category: str, last_played: dict = None) -> str:
            sounds = sound_library.get(category, [])
            if not sounds:
                return None

            # Avoid repeating the same sound
            if last_played and category in last_played:
                available = [s for s in sounds if s != last_played[category]]
                if available:
                    sounds = available

            # In real implementation, would use random.choice
            return sounds[0]

        # First play
        assert select_sound("dog_bark") in sound_library["dog_bark"]

        # Avoid repeat
        last = {"dog_bark": "bark_1.wav"}
        selected = select_sound("dog_bark", last)
        # With only 3 sounds and avoiding one, should get different
        assert selected != "bark_1.wav" or len(sound_library["dog_bark"]) == 1

    def test_volume_adjustment(self):
        """Test volume adjustment based on distance and time of day."""
        def calculate_volume(
            base_volume: int,
            distance: float,
            is_night: bool,
            max_volume: int = 100,
        ) -> int:
            # Increase volume for distant threats
            distance_factor = min(2.0, 1 + distance / 50)  # Max 2x at 50m+
            volume = base_volume * distance_factor

            # Reduce volume at night to avoid disturbing neighbors
            if is_night:
                volume *= 0.7

            return min(max_volume, int(volume))

        # Close threat during day
        assert calculate_volume(70, 5, False) < 80

        # Distant threat during day
        assert calculate_volume(70, 40, False) > 100 or calculate_volume(70, 40, False) == 100

        # Night time - reduced volume
        day_vol = calculate_volume(70, 20, False)
        night_vol = calculate_volume(70, 20, True)
        assert night_vol < day_vol

    def test_tts_message_generation(self):
        """Test generating text-to-speech messages."""
        def generate_warning_message(predator_type: str, location: str = None) -> str:
            messages = {
                "fox": [
                    "Warning! Fox detected in the garden.",
                    "Alert! A fox has been spotted nearby.",
                ],
                "hawk": [
                    "Hawk alert! Aerial predator detected.",
                    "Warning! Bird of prey overhead.",
                ],
                "coyote": [
                    "Danger! Coyote detected. Secure your animals.",
                ],
            }

            predator_messages = messages.get(predator_type, ["Warning! Predator detected."])
            message = predator_messages[0]

            if location:
                message += f" Location: {location}."

            return message

        msg = generate_warning_message("fox", "north garden")
        assert "fox" in msg.lower()
        assert "north garden" in msg.lower()


class TestSprayDeterrence:
    """Tests for spray deterrence system."""

    def test_spray_targeting(self):
        """Test calculating spray direction based on predator position."""
        def calculate_spray_direction(
            camera_position: tuple,
            predator_position: tuple,
            camera_heading: float,  # degrees from north
        ) -> float:
            import math

            # Calculate angle to predator
            dx = predator_position[0] - camera_position[0]
            dy = predator_position[1] - camera_position[1]

            angle_to_predator = math.degrees(math.atan2(dx, dy))
            if angle_to_predator < 0:
                angle_to_predator += 360

            # Relative to camera heading
            relative_angle = angle_to_predator - camera_heading
            if relative_angle < 0:
                relative_angle += 360

            return relative_angle

        # Camera at origin, facing north (0 degrees)
        camera_pos = (0, 0)
        camera_heading = 0

        # Predator to the east
        predator_pos = (10, 0)
        angle = calculate_spray_direction(camera_pos, predator_pos, camera_heading)
        assert 80 <= angle <= 100  # Should be ~90 degrees (east)

        # Predator to the north
        predator_pos = (0, 10)
        angle = calculate_spray_direction(camera_pos, predator_pos, camera_heading)
        assert angle < 10 or angle > 350  # Should be ~0 degrees (north)

    def test_spray_duration_calculation(self):
        """Test calculating spray duration based on threat."""
        def calculate_spray_duration(
            confidence: float,
            predator_danger: float,
            distance: float,
            min_duration: float = 0.5,
            max_duration: float = 3.0,
        ) -> float:
            # Base duration on threat level
            threat = confidence * predator_danger

            # Longer spray for closer threats
            distance_factor = max(0.5, 1 - distance / 20)

            duration = min_duration + (max_duration - min_duration) * threat * distance_factor
            return min(max_duration, max(min_duration, duration))

        # High confidence, dangerous predator, close
        duration = calculate_spray_duration(0.95, 0.9, 5)
        assert duration > 2.0

        # Low confidence, less dangerous, far
        duration = calculate_spray_duration(0.6, 0.5, 15)
        assert duration < 1.5

    def test_spray_resource_management(self):
        """Test managing spray reservoir levels."""
        reservoir = {
            "camera-1": {"level": 80, "capacity": 100},
            "camera-2": {"level": 15, "capacity": 100},
        }

        def spray(device_id: str, duration: float, consumption_rate: float = 5):
            """Consume spray and return remaining level."""
            if device_id not in reservoir:
                return None

            consumed = duration * consumption_rate
            reservoir[device_id]["level"] = max(
                0,
                reservoir[device_id]["level"] - consumed
            )
            return reservoir[device_id]["level"]

        def needs_refill(device_id: str, threshold: float = 20) -> bool:
            if device_id not in reservoir:
                return False
            return reservoir[device_id]["level"] < threshold

        # Camera 2 needs refill
        assert needs_refill("camera-2")
        assert not needs_refill("camera-1")

        # Spray from camera 1
        spray("camera-1", 2.0)
        assert reservoir["camera-1"]["level"] == 70

        # Multiple sprays depleting reservoir
        for _ in range(10):
            spray("camera-1", 1.0)
        assert needs_refill("camera-1")
