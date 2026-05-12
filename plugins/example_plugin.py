"""
plugins/example_plugin.py

Template for a NixOrb plugin. Drop .py files into this directory
(~/.local/share/nixorb/plugins/ or the plugins/ repo folder) and
NixOrb will load them automatically on startup or when you click
"Reload Plugins" in Settings.

The LLM will call your function automatically when the user's request
matches the description. Results are fed back into the conversation.

TOOL_DEFINITION follows the OpenAI function-calling schema.
The function name must exactly match TOOL_DEFINITION["function"]["name"].
"""
from __future__ import annotations

# ── Tool definition ────────────────────────────────────────────────── #
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Use when the user asks about weather, temperature, or rain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Berlin' or 'New York'",
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "Temperature units (default: metric)",
                },
            },
            "required": ["city"],
        },
    },
}


# ── Implementation ────────────────────────────────────────────────── #
def get_weather(city: str, units: str = "metric") -> str:
    """Fetch weather from open-meteo (no API key required)."""
    try:
        import urllib.request, json

        # 1. Geocode city
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search"
            f"?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
        )
        import urllib.parse
        with urllib.request.urlopen(geo_url, timeout=5) as r:
            geo = json.load(r)

        if not geo.get("results"):
            return f"City not found: {city}"

        loc = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        name     = loc["name"]

        # 2. Fetch current weather
        wx_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,weathercode,windspeed_10m,relativehumidity_2m"
            f"&temperature_unit={'celsius' if units=='metric' else 'fahrenheit'}"
            f"&windspeed_unit={'kmh' if units=='metric' else 'mph'}"
        )
        with urllib.request.urlopen(wx_url, timeout=5) as r:
            wx = json.load(r)

        c    = wx["current"]
        temp = c["temperature_2m"]
        unit = "°C" if units == "metric" else "°F"
        wind = c["windspeed_10m"]
        wunit = "km/h" if units == "metric" else "mph"
        hum  = c["relativehumidity_2m"]

        # WMO weather code → description
        codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
            55: "Heavy drizzle", 61: "Slight rain", 63: "Rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Snow", 75: "Heavy snow",
            80: "Showers", 81: "Rain showers", 82: "Violent showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail",
        }
        desc = codes.get(c["weathercode"], f"Code {c['weathercode']}")

        return (
            f"{name}: {desc}, {temp}{unit}, "
            f"humidity {hum}%, wind {wind} {wunit}"
        )

    except Exception as exc:
        return f"Weather fetch failed: {exc}"
