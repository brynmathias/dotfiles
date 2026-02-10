"""
Integration tests for weather integration and activity prediction.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime


class TestWeatherDataProcessing:
    """Tests for weather data processing."""

    def test_weather_data_parsing(self):
        """Test parsing weather API response."""
        api_response = {
            "main": {
                "temp": 15.5,
                "humidity": 75,
                "pressure": 1013,
            },
            "wind": {
                "speed": 5.2,
                "deg": 180,
            },
            "weather": [
                {"main": "Clouds", "description": "overcast clouds"}
            ],
            "visibility": 8000,
            "rain": {"1h": 2.5},
        }

        # Parse response
        weather_data = {
            "temperature": api_response["main"]["temp"],
            "humidity": api_response["main"]["humidity"],
            "wind_speed": api_response["wind"]["speed"],
            "wind_direction": api_response["wind"]["deg"],
            "condition": api_response["weather"][0]["main"].lower(),
            "visibility": api_response.get("visibility", 10000),
            "precipitation": api_response.get("rain", {}).get("1h", 0),
        }

        assert weather_data["temperature"] == 15.5
        assert weather_data["humidity"] == 75
        assert weather_data["wind_speed"] == 5.2
        assert weather_data["condition"] == "clouds"
        assert weather_data["precipitation"] == 2.5

    def test_weather_condition_classification(self):
        """Test classifying weather conditions."""
        def classify_condition(condition: str, visibility: int, precipitation: float) -> str:
            if precipitation > 5:
                return "heavy_rain"
            elif precipitation > 0:
                return "rain"
            elif visibility < 1000:
                return "fog"
            elif condition in ["thunderstorm", "snow"]:
                return condition
            elif condition in ["clear", "sunny"]:
                return "clear"
            else:
                return "cloudy"

        assert classify_condition("rain", 5000, 3.0) == "rain"
        assert classify_condition("rain", 5000, 10.0) == "heavy_rain"
        assert classify_condition("fog", 500, 0) == "fog"
        assert classify_condition("clear", 10000, 0) == "clear"
        assert classify_condition("clouds", 8000, 0) == "cloudy"


class TestPredatorActivityPrediction:
    """Tests for predator activity prediction based on weather."""

    def test_fox_activity_prediction(self):
        """Test predicting fox activity levels."""
        def predict_fox_activity(
            time_of_day: str,
            temperature: float,
            wind_speed: float,
            precipitation: float,
            moon_phase: float,  # 0-1, 0=new, 0.5=full
        ) -> float:
            base_activity = 0.5

            # Time of day factor (foxes are crepuscular/nocturnal)
            time_factors = {
                "dawn": 0.9,
                "day": 0.2,
                "dusk": 1.0,
                "night": 0.7,
            }
            activity = base_activity * time_factors.get(time_of_day, 0.5)

            # Temperature factor (prefer mild temps)
            if 5 <= temperature <= 20:
                activity *= 1.2
            elif temperature < 0 or temperature > 30:
                activity *= 0.6

            # Wind reduces activity
            if wind_speed > 10:
                activity *= 0.7

            # Heavy rain reduces activity
            if precipitation > 5:
                activity *= 0.3
            elif precipitation > 0:
                activity *= 0.7

            # Full moon increases night activity
            if time_of_day == "night" and moon_phase > 0.4:
                activity *= 1.1

            return min(1.0, max(0.0, activity))

        # Test dusk with good conditions
        assert predict_fox_activity("dusk", 15, 3, 0, 0.5) > 0.5

        # Test day time (low activity)
        assert predict_fox_activity("day", 15, 3, 0, 0.5) < 0.3

        # Test heavy rain (reduced activity)
        assert predict_fox_activity("dusk", 15, 3, 10, 0.5) < 0.4

    def test_hawk_activity_prediction(self):
        """Test predicting hawk activity levels."""
        def predict_hawk_activity(
            time_of_day: str,
            visibility: int,
            wind_speed: float,
            cloud_cover: float,  # 0-1
        ) -> float:
            # Hawks are diurnal and need good visibility
            if time_of_day not in ["day", "dawn", "dusk"]:
                return 0.0

            base_activity = 0.7

            # Visibility is critical for hunting
            if visibility < 2000:
                activity = base_activity * 0.2
            elif visibility < 5000:
                activity = base_activity * 0.5
            else:
                activity = base_activity

            # Moderate wind helps soaring
            if 5 <= wind_speed <= 15:
                activity *= 1.2
            elif wind_speed > 25:
                activity *= 0.5

            # Cloud cover reduces hunting (thermal activity)
            if cloud_cover > 0.8:
                activity *= 0.6

            return min(1.0, max(0.0, activity))

        # Good conditions for hawks
        assert predict_hawk_activity("day", 10000, 10, 0.3) > 0.6

        # Poor visibility
        assert predict_hawk_activity("day", 1000, 10, 0.3) < 0.3

        # Night time (no hawks)
        assert predict_hawk_activity("night", 10000, 5, 0) == 0.0

    def test_combined_threat_level(self):
        """Test calculating combined threat level from multiple predators."""
        predator_activities = {
            "fox": 0.8,
            "hawk": 0.3,
            "coyote": 0.1,
            "raccoon": 0.5,
        }

        predator_danger = {
            "fox": 0.9,
            "hawk": 0.7,
            "coyote": 1.0,
            "raccoon": 0.4,
        }

        def calculate_threat_level(activities: dict, dangers: dict) -> float:
            total_threat = 0.0
            for predator, activity in activities.items():
                danger = dangers.get(predator, 0.5)
                total_threat += activity * danger
            return min(1.0, total_threat / len(activities))

        threat = calculate_threat_level(predator_activities, predator_danger)
        assert 0 < threat < 1


class TestDetectionSensitivityAdjustment:
    """Tests for weather-based detection sensitivity adjustment."""

    def test_sensitivity_in_rain(self):
        """Test that sensitivity increases in rain to account for noise."""
        def adjust_confidence_threshold(
            base_threshold: float,
            precipitation: float,
            visibility: int,
        ) -> float:
            threshold = base_threshold

            # Increase threshold in rain (more false positives)
            if precipitation > 0:
                threshold += 0.1 * min(precipitation / 10, 1.0)

            # Increase threshold in low visibility
            if visibility < 5000:
                threshold += 0.1 * (1 - visibility / 5000)

            return min(0.95, threshold)

        base = 0.7

        # Clear conditions - no adjustment
        assert adjust_confidence_threshold(base, 0, 10000) == base

        # Light rain - small increase
        threshold_rain = adjust_confidence_threshold(base, 3, 8000)
        assert threshold_rain > base
        assert threshold_rain < 0.85

        # Heavy rain + fog - significant increase
        threshold_bad = adjust_confidence_threshold(base, 10, 1000)
        assert threshold_bad > threshold_rain

    def test_motion_threshold_adjustment(self):
        """Test adjusting motion detection threshold based on wind."""
        def adjust_motion_threshold(
            base_threshold: float,
            wind_speed: float,
        ) -> float:
            # Higher wind = more vegetation movement = higher threshold
            if wind_speed > 20:
                return base_threshold * 1.5
            elif wind_speed > 10:
                return base_threshold * 1.2
            elif wind_speed > 5:
                return base_threshold * 1.1
            return base_threshold

        base = 100  # pixels

        assert adjust_motion_threshold(base, 2) == 100
        assert adjust_motion_threshold(base, 8) == 110
        assert adjust_motion_threshold(base, 15) == 120
        assert adjust_motion_threshold(base, 25) == 150


class TestWeatherAlerts:
    """Tests for weather-related alerts."""

    def test_severe_weather_warning(self):
        """Test generating warnings for severe weather."""
        def check_severe_weather(weather_data: dict) -> list:
            warnings = []

            if weather_data.get("wind_speed", 0) > 25:
                warnings.append({
                    "type": "high_wind",
                    "severity": "warning",
                    "message": "High winds may affect detection accuracy",
                })

            if weather_data.get("temperature", 20) < -10:
                warnings.append({
                    "type": "extreme_cold",
                    "severity": "critical",
                    "message": "Risk of camera malfunction in extreme cold",
                })

            if weather_data.get("precipitation", 0) > 20:
                warnings.append({
                    "type": "heavy_precipitation",
                    "severity": "warning",
                    "message": "Heavy rain/snow reducing visibility",
                })

            return warnings

        # Normal conditions
        assert len(check_severe_weather({"wind_speed": 10, "temperature": 15})) == 0

        # High wind
        warnings = check_severe_weather({"wind_speed": 30, "temperature": 15})
        assert len(warnings) == 1
        assert warnings[0]["type"] == "high_wind"

        # Multiple warnings
        warnings = check_severe_weather({
            "wind_speed": 30,
            "temperature": -15,
            "precipitation": 25
        })
        assert len(warnings) == 3

    def test_optimal_patrol_windows(self):
        """Test identifying optimal patrol windows based on weather forecast."""
        forecast = [
            {"hour": 6, "activity": 0.8, "weather_ok": True},
            {"hour": 7, "activity": 0.7, "weather_ok": True},
            {"hour": 8, "activity": 0.3, "weather_ok": True},
            {"hour": 9, "activity": 0.2, "weather_ok": False},  # Rain
            {"hour": 10, "activity": 0.2, "weather_ok": False},
            {"hour": 17, "activity": 0.6, "weather_ok": True},
            {"hour": 18, "activity": 0.9, "weather_ok": True},
            {"hour": 19, "activity": 0.8, "weather_ok": True},
        ]

        def find_patrol_windows(forecast: list, min_activity: float = 0.5) -> list:
            windows = []
            current_window = None

            for entry in forecast:
                if entry["activity"] >= min_activity and entry["weather_ok"]:
                    if current_window is None:
                        current_window = {"start": entry["hour"], "end": entry["hour"]}
                    else:
                        current_window["end"] = entry["hour"]
                else:
                    if current_window is not None:
                        windows.append(current_window)
                        current_window = None

            if current_window is not None:
                windows.append(current_window)

            return windows

        windows = find_patrol_windows(forecast)
        assert len(windows) == 2
        assert windows[0]["start"] == 6
        assert windows[0]["end"] == 7
        assert windows[1]["start"] == 17
        assert windows[1]["end"] == 19
