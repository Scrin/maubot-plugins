"""Electricity price service for fetching current prices."""

import requests
import datetime
from typing import Dict

class ElectricityService:
    def __init__(self, vat_multiplier: float = 1.0):
        """Initialize the electricity service.
        
        Args:
            vat_multiplier: VAT multiplier to apply to prices (e.g., 1.24 for 24% VAT)
        """
        self.vat_multiplier = vat_multiplier
        self.price_cache: Dict[str, dict] = {}

    def fetch_prices(self, user: str, date: str) -> str:
        """Fetch electricity prices for a given date.
        
        Args:
            user: The username requesting the prices
            date: The date to get prices for ('today', 'tomorrow', or 'YYYY-MM-DD')
            
        Returns:
            A formatted string containing price information
        """
        # Convert 'today'/'tomorrow' to actual dates
        if date == "today":
            today = datetime.datetime.today()
            date = today.strftime('%Y-%m-%d')
        elif date == "tomorrow":
            today = datetime.datetime.today()
            tomorrow = today + datetime.timedelta(days=1)
            date = tomorrow.strftime('%Y-%m-%d')

        # Check cache
        if date in self.price_cache:
            print("Returning cached data")
            return self.price_cache[date]['data']

        # Fetch data from API
        url = f"https://www.sahkohinta-api.fi/api/v1/halpa?tunnit=24&tulos=sarja&aikaraja={date}"
        response = requests.get(url)
        
        if response.status_code != 200:
            return (f"Error: Unable to fetch data for {date} (status code {response.status_code}) "
                   "Maybe date is in the future? Prices for the next day are available around 14:00 UTC+2.")

        price_data = response.json()
        prices = [float(item['hinta']) * self.vat_multiplier for item in price_data]
        average_price = sum(prices) / len(prices)

        # Format response
        formatted_string = (
            f"Sähkön hinta on keskimäärin {average_price}. "
            "Kaikki hinnat ovat pyöristettyjä sentteinä. "
            f"Hinnat sisältävät arvonlisäveron {round(self.vat_multiplier*100-100,2)}%. "
            "Älä mainitse verotuksesta ellei erikseen kysytä. "
            "Kellonajat ovat Suomen aikaa. "
            "Vältä koko listan tulostamista käyttäjälle ja pyri kirjoittamaan kiinnostava kooste:\n"
        )

        for entry in price_data:
            price = round(float(entry['hinta']) * self.vat_multiplier, 2)
            formatted_string += f"{entry['aikaleima_suomi']}: {price} c/kWh"
            if price > average_price:
                formatted_string += " (Kalliimpi kuin keskiarvo)"
            elif price < average_price:
                formatted_string += " (Halvempi kuin keskiarvo)"
            formatted_string += "\n"

        # Cache the result
        self.price_cache[date] = {'data': formatted_string}

        return formatted_string 
