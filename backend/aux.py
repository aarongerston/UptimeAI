
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from fastapi import HTTPException
from typing import Iterable, Optional
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer


# IODA API details
IODA_API_URL = "https://api.ioda.inetintel.cc.gatech.edu/v2/"
ENTITY_TYPE = "region"  # Change as needed (could be 'country' or 'asn')

# Open-Mateo (geodata) details
OPEN_MATEO_API_URL = "https://geocoding-api.open-meteo.com/v1/search"


def _fetch_ioda_region_data(region_codes: [list, str], from_timestamp, until_timestamp):

    if isinstance(region_codes, Iterable):
        entity_code = ",".join(region_codes)
    elif isinstance(region_codes, str):
        entity_code = region_codes
    else:
        raise ValueError("Incorrect type for input parameter `region_codes`")

    params = {
        "from": from_timestamp,
        "until": until_timestamp,
        "datasource": "",
    }

    endpoint = f"signals/raw/{ENTITY_TYPE}/{entity_code}"
    try:
        response = requests.get(IODA_API_URL + endpoint, params=params)
    except requests.exceptions.ConnectTimeout:
        raise HTTPException(status_code=408, detail="Connection timeout")

    if response.status_code == 200:
        data_regions = response.json().get("data", [])
    else:
        raise HTTPException(status_code=response.status_code, detail=response.content.decode())

    return data_regions


def fetch_ioda_data(continent: Optional[str], country: Optional[str], region: Optional[str]) -> list[list[dict]]:
    """
    Fetches data from IODA API

    Step 1: fetch metadata to assess API data structure
    Step 2: use returned values to make a more detailed query for time series connectivity data

    :return: JSON result of IODA API endpoint https://api.ioda.inetintel.cc.gatech.edu/v2/signals/raw/
    """

    # Get the last 15 minutes in POSIX timestamps
    now = datetime.now()
    from_timestamp = int((now - timedelta(minutes=20)).timestamp())
    until_timestamp = int(now.timestamp())

    # Define valid regions from IODA metadata API:
    endpoint = "entities/query"
    response = requests.get(IODA_API_URL + endpoint)
    if response.status_code == 200:
        metadata = response.json()["data"]
    else:
        raise HTTPException(status_code=response.status_code, detail=response.content.decode())

    # Fetch data for each region
    continent_codes = {item["name"]: item["code"] for item in metadata if item["type"] == "continent"}
    country_codes = {item["name"]: item["code"] for item in metadata if item["type"] == "country"}

    # Filter by country/continent/region as specified by inputs
    country_code = country if country in country_codes.values() \
        else country_codes[country] if country in country_codes.keys() \
        else None
    continent_code = continent if continent in continent_codes.values() \
        else continent_codes[continent] if continent in continent_codes.keys() \
        else None
    region_codes = [
        item["code"] for item in metadata if
        item["type"] == ENTITY_TYPE and
        "Invalid" not in item["name"] and
        ((item["name"] == region) if isinstance(region, str) else True) and
        ((item["attrs"]["fqid"].split(".")[3] == country_code) if isinstance(country_code, str) else True) and
        ((item["attrs"]["fqid"].split(".")[2] == continent_code) if isinstance(continent_code, str) else True)
    ]

    data = _fetch_ioda_region_data(region_codes, from_timestamp, until_timestamp)
    data = [lst for lst in data if any(lst)]

    return data


# Function to get coordinates for a given city
def _get_city_coords(city: str, country_code: str) -> (float, float):
    """
    Uses Open-Mateo open API to fetch latitude and longitude coordinates from a given city name.

    :param city: full city name
    :param country_code: 2-letter country code
    :return: (latitude, longitude)
    """

    params = {
        "name": city,
        "language": "en",
        "format": "json",
    }
    try:
        response = requests.get(OPEN_MATEO_API_URL, params=params)
    except requests.exceptions.ConnectTimeout:  # ironically due to local connectivity issues
        return None, None

    # Parse results. If found, return them; otherwise, warn and return (None, None)
    if response.status_code == 200:
        data = response.json()
        if "results" in data.keys() and data["results"]:
            data = [result for result in data["results"] if result["country_code"] == country_code]
            if any(data):
                return data[0]["latitude"], data[0]["longitude"]

    # else:
    # warn(f"Coordinates of {city}, {country_code} not found")
    return None, None


def ioda2df(data: list[dict]):
    """
    Parses raw JSON data from IODA API https://api.ioda.inetintel.cc.gatech.edu/v2/signals/raw/
    Structures time series data as DataFrame with the following columns:
     - timestamp: pd.datetime
     - city: Full city name
     - country_code: 2-letter country code
     - [metrics...]: One column per indicator provided by IODA (expected: "merit-nt", "bgp", "ping-slash24")

    :param data:
    :return:
    """

    # Iterate through data one incrementally adding to main DataFrame
    df = pd.DataFrame(columns=("timestamp", "city", "country_code", "metric_merit-nt", "metric_bgp", "metric_ping-slash24"))
    for region_data in tqdm(data, desc="Building DataFrame", leave=False):

        # Create a temporary DataFrame for the given region to incrementally add data source to the region df
        df_region = pd.DataFrame()
        for entry in region_data:

            # Extract data
            datasource = entry["datasource"]
            start_time = datetime.fromtimestamp(entry["from"])
            step = entry["step"]
            values = entry["values"]
            city = entry["entityName"]
            fqid = entry["entityFqid"]
            country_code = fqid.split(".")[-2]  # geo.provider.[continent_code].[country_code].[region code] e.g. geo.provider.EU.ES.1007

            # Create a DataFrame for the current data source
            time_index = [start_time + timedelta(seconds=i * step) for i in range(len(values))]
            temp_df = pd.DataFrame({
                "timestamp": time_index,
                "city": city,
                "country_code": country_code,
                f"metric_{datasource}": values
            })

            # Merge with the region DataFrame
            if df_region.empty:
                df_region = temp_df
            else:
                new_columns = [col for col in temp_df.columns if col not in df_region.columns]
                df_region = pd.concat([df_region, temp_df[new_columns]], axis=1)

        # Stack the current region onto the growing main DataFrame
        df = pd.concat([df, df_region], ignore_index=True)

    # Sort df first by city then by timestamp
    df.sort_values(by=["city", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Add lat/long coordinates of cities
    unique_cities = df[["city", "country_code"]].drop_duplicates()
    iterator = tqdm(
        unique_cities.iterrows(),
        total=len(unique_cities),
        desc="Fetching geodata from Open-Mateo API",
        leave=False,
    )
    coords = {(row['city'], row['country_code']): _get_city_coords(row['city'], row['country_code']) for _, row in iterator}
    if all(v == (None, None) for v in coords.values()):
        raise HTTPException(status_code=500, detail="No coordinates found for any selected region")
    df["latitude"] = df.apply(lambda x: coords.get((x["city"], x["country_code"]), (None, None))[0], axis=1)
    df["longitude"] = df.apply(lambda x: coords.get((x["city"], x["country_code"]), (None, None))[1], axis=1)

    # Drop any regions that do not have lat/long coords, as they are meaningless to the model
    df.dropna(subset=['longitude', 'latitude'], inplace=True)

    return df


# Placeholder black-box model
def predict_outages(df: pd.DataFrame, model_path: str) -> dict:
    """
    Placeholder function to return outage likelihood per region

    :param df:

    :return: dict {region (str): prediction (float)}
    """

    # Simple Imputer
    imputer = SimpleImputer(strategy="mean")
    features = [c for c in df.columns if c.startswith(("metric_", "feature_"))]
    nan_columns = df[features].isna().all()
    valid_features = nan_columns[~nan_columns].index.tolist()  # Columns that remain after imputation
    imputed = imputer.fit_transform(df[valid_features])
    X = pd.DataFrame(imputed, columns=valid_features, index=df.index)  # Create DataFrame with only the imputed columns
    for col in nan_columns[nan_columns].index:  # Re-add the entirely NaN columns, ensuring correct order
        X[col] = np.nan

    # Make sure X's columns are in the correct order
    column_order = [
        'metric_merit-nt',
        'metric_bgp',
        'metric_ping-slash24',
        'feature_calc_merit-nt_mad_lag_5m_nearby',
        'feature_calc_bgp_mad_lag_5m_nearby',
        'feature_calc_ping-slash24_mad_lag_5m_nearby',
        'feature_calc_avg_mad_lag_5m_nearby',
        'feature_calc_merit-nt_mad_lag_10m_nearby',
        'feature_calc_bgp_mad_lag_10m_nearby',
        'feature_calc_ping-slash24_mad_lag_10m_nearby',
        'feature_calc_avg_mad_lag_10m_nearby',
        'feature_calc_merit-nt_mad_lag_15m_nearby',
        'feature_calc_bgp_mad_lag_15m_nearby',
        'feature_calc_ping-slash24_mad_lag_15m_nearby',
        'feature_calc_avg_mad_lag_15m_nearby',
    ]
    if any(col not in X.columns for col in column_order):
        raise Exception(422, "Model inputs not satisfied")
    X = X[column_order]

    # Load model
    vae = tf.keras.models.load_model(model_path)

    # Make predictions
    reconstructed_X = vae.predict(X)
    recons_error = np.mean((X - reconstructed_X) ** 2, axis=1)

    # Min-Max scale (to output 0-1)
    predictions = (recons_error - recons_error.min()) / (
        recons_error.max() - recons_error.min()
    )

    # Output as dict
    predictions = {region: prediction for region, prediction in zip(df["city"], predictions)}

    return predictions