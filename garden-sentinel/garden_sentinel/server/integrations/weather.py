"""
Weather integration for Garden Sentinel.

Provides:
- Current weather conditions
- Forecasts
- Predator activity predictions based on weather
- Detection sensitivity adjustments
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable
import aiohttp

logger = logging.getLogger(__name__)


class WeatherCondition(Enum):
    """Weather condition categories."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    FOG = "fog"
    STORM = "storm"
    WINDY = "windy"


class TimeOfDay(Enum):
    """Time of day categories."""
    DAWN = "dawn"       # 1 hour before sunrise to 1 hour after
    DAY = "day"
    DUSK = "dusk"       # 1 hour before sunset to 1 hour after
    NIGHT = "night"


@dataclass
class WeatherData:
    """Current weather conditions."""
    timestamp: float
    temperature_c: float
    feels_like_c: float
    humidity: float  # 0-100
    pressure_hpa: float
    wind_speed_ms: float
    wind_direction: int  # degrees
    cloud_cover: float  # 0-100
    visibility_m: float
    condition: WeatherCondition
    description: str

    # Precipitation
    rain_1h_mm: float = 0.0
    snow_1h_mm: float = 0.0

    # Sun times
    sunrise: Optional[float] = None
    sunset: Optional[float] = None

    # Moon phase (0-1, 0=new, 0.5=full)
    moon_phase: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "temperature_c": self.temperature_c,
            "feels_like_c": self.feels_like_c,
            "humidity": self.humidity,
            "wind_speed_ms": self.wind_speed_ms,
            "condition": self.condition.value,
            "description": self.description,
            "visibility_m": self.visibility_m,
            "rain_1h_mm": self.rain_1h_mm,
        }


@dataclass
class WeatherForecast:
    """Weather forecast for a specific time."""
    timestamp: float
    temperature_c: float
    condition: WeatherCondition
    precipitation_probability: float  # 0-1
    wind_speed_ms: float


@dataclass
class PredatorActivityForecast:
    """Predicted predator activity based on conditions."""
    time_of_day: TimeOfDay
    weather: WeatherCondition
    activity_level: float  # 0-1, higher = more active
    risk_factors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class WeatherProvider:
    """Base class for weather data providers."""

    async def get_current(self, lat: float, lon: float) -> Optional[WeatherData]:
        raise NotImplementedError

    async def get_forecast(
        self,
        lat: float,
        lon: float,
        hours: int = 24,
    ) -> list[WeatherForecast]:
        raise NotImplementedError


class OpenWeatherMapProvider(WeatherProvider):
    """
    Weather data from OpenWeatherMap API.

    Free tier: 1000 calls/day
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session:
            await self._session.close()

    def _parse_condition(self, weather_id: int) -> WeatherCondition:
        """Convert OpenWeatherMap weather ID to condition."""
        if weather_id < 300:  # Thunderstorm
            return WeatherCondition.STORM
        elif weather_id < 400:  # Drizzle
            return WeatherCondition.RAIN
        elif weather_id < 600:  # Rain
            if weather_id >= 502:
                return WeatherCondition.HEAVY_RAIN
            return WeatherCondition.RAIN
        elif weather_id < 700:  # Snow
            return WeatherCondition.SNOW
        elif weather_id < 800:  # Atmosphere (fog, mist, etc.)
            return WeatherCondition.FOG
        elif weather_id == 800:  # Clear
            return WeatherCondition.CLEAR
        else:  # Clouds
            return WeatherCondition.CLOUDY

    async def get_current(self, lat: float, lon: float) -> Optional[WeatherData]:
        """Get current weather conditions."""
        session = await self._get_session()

        try:
            url = f"{self.BASE_URL}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
            }

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Weather API error: {response.status}")
                    return None

                data = await response.json()

            weather = data.get("weather", [{}])[0]
            main = data.get("main", {})
            wind = data.get("wind", {})
            clouds = data.get("clouds", {})
            rain = data.get("rain", {})
            snow = data.get("snow", {})
            sys = data.get("sys", {})

            return WeatherData(
                timestamp=data.get("dt", time.time()),
                temperature_c=main.get("temp", 0),
                feels_like_c=main.get("feels_like", 0),
                humidity=main.get("humidity", 0),
                pressure_hpa=main.get("pressure", 1013),
                wind_speed_ms=wind.get("speed", 0),
                wind_direction=wind.get("deg", 0),
                cloud_cover=clouds.get("all", 0),
                visibility_m=data.get("visibility", 10000),
                condition=self._parse_condition(weather.get("id", 800)),
                description=weather.get("description", ""),
                rain_1h_mm=rain.get("1h", 0),
                snow_1h_mm=snow.get("1h", 0),
                sunrise=sys.get("sunrise"),
                sunset=sys.get("sunset"),
            )

        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return None

    async def get_forecast(
        self,
        lat: float,
        lon: float,
        hours: int = 24,
    ) -> list[WeatherForecast]:
        """Get weather forecast."""
        session = await self._get_session()

        try:
            url = f"{self.BASE_URL}/forecast"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
                "cnt": hours // 3 + 1,  # API returns 3-hour intervals
            }

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []

                data = await response.json()

            forecasts = []
            for item in data.get("list", []):
                weather = item.get("weather", [{}])[0]
                main = item.get("main", {})
                wind = item.get("wind", {})
                pop = item.get("pop", 0)  # Probability of precipitation

                forecasts.append(WeatherForecast(
                    timestamp=item.get("dt", 0),
                    temperature_c=main.get("temp", 0),
                    condition=self._parse_condition(weather.get("id", 800)),
                    precipitation_probability=pop,
                    wind_speed_ms=wind.get("speed", 0),
                ))

            return forecasts

        except Exception as e:
            logger.error(f"Error fetching forecast: {e}")
            return []


class OpenMeteoProvider(WeatherProvider):
    """
    Weather data from Open-Meteo API.

    Free, no API key required!
    """

    BASE_URL = "https://api.open-meteo.com/v1"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session:
            await self._session.close()

    def _parse_wmo_code(self, code: int) -> WeatherCondition:
        """Convert WMO weather code to condition."""
        if code <= 1:
            return WeatherCondition.CLEAR
        elif code <= 3:
            return WeatherCondition.CLOUDY
        elif code <= 49:
            return WeatherCondition.FOG
        elif code <= 59:
            return WeatherCondition.RAIN
        elif code <= 69:
            return WeatherCondition.SNOW
        elif code <= 79:
            return WeatherCondition.RAIN
        elif code <= 84:
            return WeatherCondition.RAIN
        elif code <= 94:
            return WeatherCondition.HEAVY_RAIN
        else:
            return WeatherCondition.STORM

    async def get_current(self, lat: float, lon: float) -> Optional[WeatherData]:
        """Get current weather conditions."""
        session = await self._get_session()

        try:
            url = f"{self.BASE_URL}/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                          "precipitation,rain,weather_code,cloud_cover,pressure_msl,"
                          "wind_speed_10m,wind_direction_10m",
                "daily": "sunrise,sunset",
                "timezone": "auto",
            }

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Weather API error: {response.status}")
                    return None

                data = await response.json()

            current = data.get("current", {})
            daily = data.get("daily", {})

            weather_code = current.get("weather_code", 0)

            return WeatherData(
                timestamp=time.time(),
                temperature_c=current.get("temperature_2m", 0),
                feels_like_c=current.get("apparent_temperature", 0),
                humidity=current.get("relative_humidity_2m", 0),
                pressure_hpa=current.get("pressure_msl", 1013),
                wind_speed_ms=current.get("wind_speed_10m", 0) / 3.6,  # km/h to m/s
                wind_direction=current.get("wind_direction_10m", 0),
                cloud_cover=current.get("cloud_cover", 0),
                visibility_m=10000,  # Not provided
                condition=self._parse_wmo_code(weather_code),
                description=f"WMO code: {weather_code}",
                rain_1h_mm=current.get("rain", 0),
            )

        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return None

    async def get_forecast(
        self,
        lat: float,
        lon: float,
        hours: int = 24,
    ) -> list[WeatherForecast]:
        """Get weather forecast."""
        session = await self._get_session()

        try:
            url = f"{self.BASE_URL}/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,weather_code,precipitation_probability,wind_speed_10m",
                "timezone": "auto",
                "forecast_hours": hours,
            }

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []

                data = await response.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            codes = hourly.get("weather_code", [])
            probs = hourly.get("precipitation_probability", [])
            winds = hourly.get("wind_speed_10m", [])

            forecasts = []
            for i, t in enumerate(times[:hours]):
                dt = datetime.fromisoformat(t)
                forecasts.append(WeatherForecast(
                    timestamp=dt.timestamp(),
                    temperature_c=temps[i] if i < len(temps) else 0,
                    condition=self._parse_wmo_code(codes[i] if i < len(codes) else 0),
                    precipitation_probability=(probs[i] / 100) if i < len(probs) else 0,
                    wind_speed_ms=(winds[i] / 3.6) if i < len(winds) else 0,
                ))

            return forecasts

        except Exception as e:
            logger.error(f"Error fetching forecast: {e}")
            return []


class PredatorActivityPredictor:
    """
    Predicts predator activity levels based on weather and time.

    Different predators have different activity patterns:
    - Foxes: Most active at dawn/dusk, avoid heavy rain
    - Badgers: Nocturnal, less active in cold
    - Birds of prey: Active during day, affected by visibility
    """

    # Activity modifiers by predator type
    PREDATOR_PATTERNS = {
        "fox": {
            "time_weights": {
                TimeOfDay.DAWN: 1.0,
                TimeOfDay.DAY: 0.3,
                TimeOfDay.DUSK: 1.0,
                TimeOfDay.NIGHT: 0.7,
            },
            "weather_weights": {
                WeatherCondition.CLEAR: 0.9,
                WeatherCondition.CLOUDY: 1.0,
                WeatherCondition.RAIN: 0.5,
                WeatherCondition.HEAVY_RAIN: 0.2,
                WeatherCondition.FOG: 0.8,
                WeatherCondition.STORM: 0.1,
            },
            "temp_range": (-5, 30),  # Comfortable range
            "moon_sensitive": True,  # More active during full moon
        },
        "badger": {
            "time_weights": {
                TimeOfDay.DAWN: 0.5,
                TimeOfDay.DAY: 0.1,
                TimeOfDay.DUSK: 0.8,
                TimeOfDay.NIGHT: 1.0,
            },
            "weather_weights": {
                WeatherCondition.CLEAR: 0.9,
                WeatherCondition.CLOUDY: 1.0,
                WeatherCondition.RAIN: 0.6,
                WeatherCondition.HEAVY_RAIN: 0.3,
                WeatherCondition.FOG: 0.9,
                WeatherCondition.STORM: 0.2,
            },
            "temp_range": (5, 25),
            "moon_sensitive": False,
        },
        "bird_of_prey": {
            "time_weights": {
                TimeOfDay.DAWN: 0.9,
                TimeOfDay.DAY: 1.0,
                TimeOfDay.DUSK: 0.8,
                TimeOfDay.NIGHT: 0.1,
            },
            "weather_weights": {
                WeatherCondition.CLEAR: 1.0,
                WeatherCondition.CLOUDY: 0.8,
                WeatherCondition.RAIN: 0.3,
                WeatherCondition.HEAVY_RAIN: 0.1,
                WeatherCondition.FOG: 0.2,
                WeatherCondition.STORM: 0.0,
            },
            "temp_range": (0, 35),
            "moon_sensitive": False,
        },
    }

    def get_time_of_day(
        self,
        timestamp: float,
        sunrise: Optional[float] = None,
        sunset: Optional[float] = None,
    ) -> TimeOfDay:
        """Determine time of day category."""
        if sunrise is None or sunset is None:
            # Approximate based on hour
            hour = datetime.fromtimestamp(timestamp).hour
            if 5 <= hour < 8:
                return TimeOfDay.DAWN
            elif 8 <= hour < 17:
                return TimeOfDay.DAY
            elif 17 <= hour < 20:
                return TimeOfDay.DUSK
            else:
                return TimeOfDay.NIGHT

        # Use actual sunrise/sunset
        dawn_start = sunrise - 3600
        dawn_end = sunrise + 3600
        dusk_start = sunset - 3600
        dusk_end = sunset + 3600

        if dawn_start <= timestamp <= dawn_end:
            return TimeOfDay.DAWN
        elif dawn_end < timestamp < dusk_start:
            return TimeOfDay.DAY
        elif dusk_start <= timestamp <= dusk_end:
            return TimeOfDay.DUSK
        else:
            return TimeOfDay.NIGHT

    def predict_activity(
        self,
        predator_type: str,
        weather: WeatherData,
        timestamp: Optional[float] = None,
    ) -> PredatorActivityForecast:
        """Predict activity level for a specific predator type."""
        timestamp = timestamp or time.time()

        patterns = self.PREDATOR_PATTERNS.get(
            predator_type,
            self.PREDATOR_PATTERNS["fox"],  # Default to fox
        )

        # Get time of day
        time_of_day = self.get_time_of_day(
            timestamp,
            weather.sunrise,
            weather.sunset,
        )

        # Calculate base activity
        time_weight = patterns["time_weights"].get(time_of_day, 0.5)
        weather_weight = patterns["weather_weights"].get(weather.condition, 0.5)

        # Temperature modifier
        temp_min, temp_max = patterns["temp_range"]
        if temp_min <= weather.temperature_c <= temp_max:
            temp_modifier = 1.0
        elif weather.temperature_c < temp_min:
            temp_modifier = max(0.3, 1 - (temp_min - weather.temperature_c) / 20)
        else:
            temp_modifier = max(0.3, 1 - (weather.temperature_c - temp_max) / 20)

        # Moon modifier
        moon_modifier = 1.0
        if patterns.get("moon_sensitive") and weather.moon_phase is not None:
            # More active around full moon
            moon_modifier = 0.8 + 0.4 * (1 - abs(0.5 - weather.moon_phase) * 2)

        # Wind modifier (high wind reduces activity)
        wind_modifier = max(0.5, 1 - weather.wind_speed_ms / 20)

        # Calculate final activity level
        activity = (
            time_weight * 0.35 +
            weather_weight * 0.30 +
            temp_modifier * 0.15 +
            moon_modifier * 0.10 +
            wind_modifier * 0.10
        )

        activity = min(1.0, max(0.0, activity))

        # Build risk factors and recommendations
        risk_factors = []
        recommendations = []

        if time_weight > 0.7:
            risk_factors.append(f"High activity time ({time_of_day.value})")

        if weather.condition in (WeatherCondition.FOG, WeatherCondition.CLOUDY):
            risk_factors.append("Reduced visibility may embolden predators")

        if weather.temperature_c < temp_min:
            recommendations.append("Cold weather - predators may be seeking food")

        if activity > 0.7:
            recommendations.append("Consider increasing patrol frequency")
            recommendations.append("Ensure all deterrents are active")
        elif activity < 0.3:
            recommendations.append("Low activity expected - routine monitoring sufficient")

        return PredatorActivityForecast(
            time_of_day=time_of_day,
            weather=weather.condition,
            activity_level=activity,
            risk_factors=risk_factors,
            recommendations=recommendations,
        )

    def get_detection_sensitivity_adjustment(
        self,
        weather: WeatherData,
    ) -> dict:
        """
        Recommend detection sensitivity adjustments based on weather.

        Returns adjustment factors for detection parameters.
        """
        adjustments = {
            "confidence_threshold": 0.0,  # Adjust threshold
            "motion_sensitivity": 0.0,  # Adjust motion detection
            "frame_skip": 0,  # Skip more/fewer frames
        }

        # Low visibility - lower confidence threshold to catch more
        if weather.visibility_m < 1000:
            adjustments["confidence_threshold"] = -0.1
            adjustments["motion_sensitivity"] = 0.2

        # Rain creates motion noise - increase threshold
        if weather.condition in (WeatherCondition.RAIN, WeatherCondition.HEAVY_RAIN):
            adjustments["confidence_threshold"] = 0.1
            adjustments["motion_sensitivity"] = -0.3

        # Fog - increase sensitivity
        if weather.condition == WeatherCondition.FOG:
            adjustments["confidence_threshold"] = -0.05
            adjustments["motion_sensitivity"] = 0.1

        # High wind - motion noise
        if weather.wind_speed_ms > 10:
            adjustments["motion_sensitivity"] = -0.2

        return adjustments


class WeatherService:
    """
    Main weather service integrating providers and predictions.
    """

    def __init__(
        self,
        provider: WeatherProvider,
        latitude: float,
        longitude: float,
        update_interval: float = 900,  # 15 minutes
    ):
        self.provider = provider
        self.latitude = latitude
        self.longitude = longitude
        self.update_interval = update_interval

        self.predictor = PredatorActivityPredictor()

        self._current_weather: Optional[WeatherData] = None
        self._forecast: list[WeatherForecast] = []
        self._last_update: float = 0
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        # Callbacks
        self._weather_callbacks: list[Callable] = []

    def add_weather_callback(self, callback: Callable[[WeatherData], None]):
        """Add callback for weather updates."""
        self._weather_callbacks.append(callback)

    @property
    def current(self) -> Optional[WeatherData]:
        """Get current weather data."""
        return self._current_weather

    @property
    def forecast(self) -> list[WeatherForecast]:
        """Get weather forecast."""
        return self._forecast

    async def update(self):
        """Fetch latest weather data."""
        weather = await self.provider.get_current(self.latitude, self.longitude)

        if weather:
            self._current_weather = weather
            self._last_update = time.time()

            # Notify callbacks
            for callback in self._weather_callbacks:
                try:
                    callback(weather)
                except Exception as e:
                    logger.error(f"Weather callback error: {e}")

        # Update forecast
        self._forecast = await self.provider.get_forecast(
            self.latitude, self.longitude, hours=24
        )

    async def _update_loop(self):
        """Background update loop."""
        while self._running:
            try:
                await self.update()
            except Exception as e:
                logger.error(f"Weather update error: {e}")

            await asyncio.sleep(self.update_interval)

    async def start(self):
        """Start the weather service."""
        self._running = True
        await self.update()  # Initial fetch
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Weather service started")

    async def stop(self):
        """Stop the weather service."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        await self.provider.close()
        logger.info("Weather service stopped")

    def get_activity_prediction(
        self,
        predator_type: str = "fox",
    ) -> Optional[PredatorActivityForecast]:
        """Get current activity prediction."""
        if not self._current_weather:
            return None

        return self.predictor.predict_activity(
            predator_type,
            self._current_weather,
        )

    def get_detection_adjustments(self) -> dict:
        """Get recommended detection parameter adjustments."""
        if not self._current_weather:
            return {}

        return self.predictor.get_detection_sensitivity_adjustment(
            self._current_weather
        )
