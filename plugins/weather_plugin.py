"""plugins/weather_plugin.py — Live weather via open-meteo (no API key)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get the current weather for any city. "
            "Use when the user asks about temperature, rain, wind, or conditions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Berlin' or 'San Francisco'",
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "Temperature units. Default: metric (Celsius).",
                },
            },
            "required": ["city"],
        },
    },
}

_WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Showers", 81: "Rain showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm + hail",
}


def get_weather(city: str, units: str = "metric") -> str:
    try:
        # Geocode
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": city, "count": 1, "language": "en", "format": "json"})
        )
        with urllib.request.urlopen(geo_url, timeout=6) as r:
            geo = json.load(r)

        if not geo.get("results"):
            return f"City not found: {city}"

        loc  = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        name = loc["name"]
        country = loc.get("country", "")

        # Weather
        celsius = "true" if units == "imperial" else "false"
        wx_url  = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,weathercode,windspeed_10m,relativehumidity_2m"
            f"&temperature_unit={'fahrenheit' if units == 'imperial' else 'celsius'}"
            f"&windspeed_unit={'mph' if units == 'imperial' else 'kmh'}"
        )
        with urllib.request.urlopen(wx_url, timeout=6) as r:
            wx = json.load(r)

        c     = wx["current"]
        temp  = c["temperature_2m"]
        unit  = "°F" if units == "imperial" else "°C"
        wind  = c["windspeed_10m"]
        wunit = "mph" if units == "imperial" else "km/h"
        hum   = c["relativehumidity_2m"]
        desc  = _WMO.get(c["weathercode"], f"Code {c['weathercode']}")

        return (
            f"{name}, {country}: {desc} · {temp}{unit} · "
            f"humidity {hum}% · wind {wind} {wunit}"
        )
    except Exception as exc:
        return f"Weather error: {exc}"
