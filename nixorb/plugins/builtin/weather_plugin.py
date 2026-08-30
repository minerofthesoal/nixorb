"""
nixorb/plugins/builtin/weather_plugin.py

Built-in plugin: current weather and short-term forecast for a place name.

Uses Open-Meteo (https://open-meteo.com) — free, no API key, no signup.
Two calls: geocoding (place name -> lat/lon) then the forecast endpoint.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get the current weather and today's forecast for a place. "
            "Accepts a city name, 'City, Country', or a landmark, e.g. "
            "'San Jose', 'Paris, France', 'Tokyo'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Place name to look up.",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units (default: celsius).",
                },
            },
            "required": ["location"],
        },
    },
}

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 10

# WMO weather interpretation codes -> short description.
_WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def _get_json(url: str, params: dict) -> dict:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full_url, headers={"User-Agent": "nixorb/weather-plugin"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_weather(location: str, units: str = "celsius") -> str:
    try:
        geo = _get_json(_GEOCODE_URL, {"name": location, "count": 1, "language": "en", "format": "json"})
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return f"Couldn't reach the geocoding service: {exc}"

    results = geo.get("results") or []
    if not results:
        return f"Couldn't find a place called '{location}'."

    place = results[0]
    lat, lon = place["latitude"], place["longitude"]
    label_parts = [place.get("name", location)]
    if place.get("admin1"):
        label_parts.append(place["admin1"])
    if place.get("country"):
        label_parts.append(place["country"])
    label = ", ".join(label_parts)

    temp_unit = "fahrenheit" if units == "fahrenheit" else "celsius"
    try:
        wx = _get_json(_FORECAST_URL, {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                       "weather_code,wind_speed_10m,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "temperature_unit": temp_unit,
            "wind_speed_unit": "mph" if temp_unit == "fahrenheit" else "kmh",
            "timezone": "auto",
            "forecast_days": 1,
        })
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return f"Couldn't reach the forecast service: {exc}"

    cur = wx.get("current", {})
    daily = wx.get("daily", {})
    deg = "°F" if temp_unit == "fahrenheit" else "°C"
    speed_unit = "mph" if temp_unit == "fahrenheit" else "km/h"
    condition = _WMO_CODES.get(cur.get("weather_code"), "unknown conditions")

    parts = [f"{label}: {condition}, {cur.get('temperature_2m')}{deg}"]
    if "apparent_temperature" in cur:
        parts.append(f"(feels like {cur['apparent_temperature']}{deg})")
    if "relative_humidity_2m" in cur:
        parts.append(f"· humidity {cur['relative_humidity_2m']}%")
    if "wind_speed_10m" in cur:
        parts.append(f"· wind {cur['wind_speed_10m']} {speed_unit}")
    summary = " ".join(parts)

    if daily.get("temperature_2m_max") and daily.get("temperature_2m_min"):
        summary += (
            f"\nToday: high {daily['temperature_2m_max'][0]}{deg}, "
            f"low {daily['temperature_2m_min'][0]}{deg}"
        )
        if daily.get("precipitation_probability_max"):
            summary += f", {daily['precipitation_probability_max'][0]}% chance of precipitation"

    return summary
