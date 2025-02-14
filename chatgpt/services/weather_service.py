"""Weather service for fetching weather data."""

import fmi_weather_client as fmi
from fmi_weather_client.errors import ClientError, ServerError
from ..constants.weather_map import WEATHER_MAP

def get_weather(user: str, location: str) -> str:
    """Get current and forecasted weather for a location.
    
    Args:
        user: The username requesting the weather
        location: The location to get weather for
        
    Returns:
        A formatted string containing current and forecasted weather
    """
    try:
        current_data = fmi.weather_by_place_name(location)
    except ClientError as err:
        print(f"Client error with status {err.status_code}: {err.message}")
        return f"Error fetching current weather: {err.message}"
    except ServerError as err:
        print(f"Server error with status {err.status_code}: {err.body}")
        return f"Error fetching current weather: {err.body}"

    try:
        forecast_data = fmi.forecast_by_place_name(location, timestep_hours=6)
    except ClientError as err:
        print(f"Client error with status {err.status_code}: {err.message}")
        return f"Error fetching forecast: {err.message}"
    except ServerError as err:
        print(f"Server error with status {err.status_code}: {err.body}")
        return f"Error fetching forecast: {err.body}"

    # Extract current weather data
    current_temperature = current_data.data.temperature
    current_feels_like = current_data.data.feels_like
    current_humidity = current_data.data.humidity
    try:
        symbol_value = current_data.data.symbol.value
        current_weather = WEATHER_MAP[symbol_value if symbol_value < 100 else symbol_value - 100]
    except:
        current_weather = "Tuntematon"
    current_wind_speed = current_data.data.wind_speed
    current_wind_deg = current_data.data.wind_direction
    current_gust_speed = current_data.data.wind_gust
    current_clouds = current_data.data.cloud_cover
    current_pressure = current_data.data.pressure
    current_precipitation = current_data.data.precipitation_amount

    # Format current weather
    current_weather_str = (
        f"Tämänhetkinen sää: {current_temperature.value}{current_temperature.unit} "
        f"(Tuntuu kuin {current_feels_like.value:2f}{current_feels_like.unit}), "
        f"{current_weather}, {current_humidity.value}{current_humidity.unit} ilmankosteus, "
        f"tuulen nopeus: {current_wind_speed.value}{current_wind_speed.unit}, "
        f"tuulen suunta {current_wind_deg.value}{current_wind_deg.unit} "
        f"(Puuskissa {current_gust_speed.value}{current_gust_speed.unit}), "
        f"Pilvisyys: {current_clouds.value}{current_clouds.unit}, "
        f"Sademäärä: {current_precipitation.value}{current_precipitation.unit}, "
        f"ilmanpaine {current_pressure.value}{current_precipitation.unit}"
    )

    # Format forecast weather
    forecasted_weather_str = "Ennustettu sää:\n"
    for forecast in forecast_data.forecasts:
        forecast_time = forecast.time
        forecast_temperature = forecast.temperature
        forecast_humidity = forecast.humidity
        try:
            symbol_value = forecast.symbol.value
            forecast_weather = WEATHER_MAP[symbol_value if symbol_value < 100 else symbol_value - 100]
        except:
            forecast_weather = "Tuntematon"
        forecast_wind_speed = forecast.wind_speed
        forecast_precipitation = forecast.precipitation_amount

        weekday = forecast_time.strftime("%A")
        forecasted_weather_str += (
            f"{weekday} - {forecast_time}: {forecast_temperature.value}{forecast_temperature.unit}, "
            f"{forecast_weather}, {forecast_humidity.value}{forecast_humidity.unit} ilmankosteus, "
            f"Tuulen nopeus: {forecast_wind_speed.value}{forecast_wind_speed.unit}, "
            f"Sademäärä: {forecast_precipitation.value}{forecast_precipitation.unit}\n"
        )

    return f"{current_weather_str}\n\n{forecasted_weather_str}" 
