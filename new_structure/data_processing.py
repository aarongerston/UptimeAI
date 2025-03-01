"""
data_processing.py

Responsible for merging raw data into a Pandas DataFrame,
and adding geocoding information (city latitude & longitude).
Also includes helper functions for distance-based lookups.
"""

import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from geopy.distance import great_circle
from warnings import warn
from typing import Tuple, Optional

from config import logger


def get_city_coords(
    city: str, country_code: str
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get geographic coordinates (latitude, longitude) for a given city
    via the Open-Meteo geocoding API.
    Returns (None, None) if not found.

    :param city: City name
    :param country_code: ISO country code (e.g., "ES" for Spain)
    :return: (latitude, longitude) or (None, None)
    """
    base_url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "language": "en",
        "format": "json",
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.warning(f"Geocoding request failed for {city}, {country_code}: {e}")
        return None, None

    if "results" in data and data["results"]:
        # Filter by matching country code
        results = [r for r in data["results"] if r.get("country_code") == country_code]
        if results:
            return results[0]["latitude"], results[0]["longitude"]

    warn(f"Coordinates of {city}, {country_code} not found.")
    return None, None


def merge_sources_into_df(data: list) -> pd.DataFrame:
    """
    Merge multiple source signals from the IODA response into a single DataFrame.
    Also fetch city coordinates for each row.

    :param data: List of lists/dicts, each containing 'datasource', 'from', 'step', 'values', 'entityName'
    :return: A DataFrame with columns [timestamp, city, country_code, metric_*, latitude, longitude]
    """
    df = pd.DataFrame()
    for region_data in data:  # data is typically a list of lists by region
        df_region = pd.DataFrame()
        for entry in region_data:
            datasource = entry["datasource"]
            start_time = datetime.fromtimestamp(entry["from"])
            step = entry["step"]
            values = entry["values"]
            city = entry["entityName"]
            fqid = entry["entityFqid"]
            # e.g. geo.provider.EU.ES.1007 => the second to last segment is the country code
            split_fqid = fqid.split(".")
            if len(split_fqid) >= 2:
                country_code = split_fqid[-2].upper()
            else:
                country_code = "XX"

            # Create time index
            time_index = [
                start_time + timedelta(seconds=i * step) for i in range(len(values))
            ]

            temp_df = pd.DataFrame(
                {
                    "timestamp": time_index,
                    "city": city,
                    "country_code": country_code,
                    f"metric_{datasource}": values,
                }
            )

            # Merge columns by horizontal concatenation
            if df_region.empty:
                df_region = temp_df
            else:
                new_columns = [
                    col for col in temp_df.columns if col not in df_region.columns
                ]
                # If the length mismatch occurs, we might have to do a true merge on timestamp
                if len(temp_df) == len(df_region):
                    df_region = pd.concat([df_region, temp_df[new_columns]], axis=1)
                else:
                    # Fallback: merge on timestamp (if they share timestamps)
                    df_region = pd.merge(
                        df_region,
                        temp_df,
                        on=["timestamp", "city", "country_code"],
                        how="outer",
                    )

        df = pd.concat([df, df_region], ignore_index=True)

    # Sort and reset index
    df.sort_values(by=["city", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Collect unique (city, country_code) pairs
    unique_cities = df[["city", "country_code"]].drop_duplicates()

    # Build a lookup of coordinates
    coords = {}
    for _, row in unique_cities.iterrows():
        lat, lon = get_city_coords(row["city"], row["country_code"])
        coords[(row["city"], row["country_code"])] = (lat, lon)

    # Apply coordinate mapping
    df["latitude"] = df.apply(
        lambda x: coords.get((x["city"], x["country_code"]), (None, None))[0], axis=1
    )
    df["longitude"] = df.apply(
        lambda x: coords.get((x["city"], x["country_code"]), (None, None))[1], axis=1
    )

    # Drop rows with no location info
    df.dropna(subset=["latitude", "longitude"], inplace=True)

    return df


def find_nearby_cities(
    df: pd.DataFrame, target_city: str, radius_km: float = 100
) -> np.ndarray:
    """
    Given a DataFrame containing city-level data with latitude and longitude,
    return an array of nearby city names within a given radius from the target_city.

    :param df: DataFrame with columns ['city', 'latitude', 'longitude']
    :param target_city: The city for which we want neighbors
    :param radius_km: Distance threshold in kilometers
    :return: Numpy array of nearby city names
    """
    target_row = df[df["city"] == target_city]
    if target_row.empty:
        logger.warning(
            f"No valid coordinates found for {target_city}. Returning empty list."
        )
        return np.array([])

    target_coords = target_row[["latitude", "longitude"]].values[0]

    def within_radius(row):
        dist = great_circle(target_coords, (row["latitude"], row["longitude"])).km
        return dist <= radius_km

    mask = df.apply(within_radius, axis=1)
    return df.loc[mask, "city"].unique()
