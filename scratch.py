import numpy as np
import requests
import pandas as pd
import seaborn as sns
from warnings import warn
import matplotlib.pyplot as plt
from geopy.distance import great_circle
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from tensorflow.keras import layers, models, Input


# IODA API Endpoint
API_URL = "https://api.ioda.inetintel.cc.gatech.edu/v2/"


# Function to get coordinates for a given city
def get_city_coords(city: str, country_code: str):

    base_url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city,
        "language": "en",
        "format": "json",
    }
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if "results" in data.keys() and data["results"]:
            data = [result for result in data["results"] if result["country_code"] == country_code]
            if any(data):
                return data[0]["latitude"], data[0]["longitude"]

    # else:
    warn(f"Coordinates of {city}, {country} not found")
    return None, None


def merge_sources_into_df(data: list[dict]):

    df = pd.DataFrame()
    for region_data in data:

        df_region = pd.DataFrame()
        for entry in region_data:

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

            # Merge with the main DataFrame
            if df_region.empty:
                df_region = temp_df
            else:
                new_columns = [col for col in temp_df.columns if col not in df_region.columns]
                df_region = pd.concat([df_region, temp_df[new_columns]], axis=1)

        # Stack the current region onto the growing df DataFrame
        df = pd.concat([df, df_region], ignore_index=True)

    # Convert dictionary to DataFrame
    df.sort_values(by=["timestamp", "city"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # df.fillna(0, inplace=True)  # Replace missing values with 0

    # Add xy coordinates of cities
    unique_cities = df[["city", "country_code"]].drop_duplicates()
    coords = {
        (row['city'], row['country_code']): get_city_coords(row['city'], row['country_code'])
        for _, row in unique_cities.iterrows()
    }
    # Map coordinates to the dataframe
    df["latitude"] = df.apply(lambda x: coords.get((x["city"], x["country_code"]), (None, None))[0], axis=1)
    df["longitude"] = df.apply(lambda x: coords.get((x["city"], x["country_code"]), (None, None))[1], axis=1)

    return df


def find_nearby_cities(df: pd.DataFrame, target_city: str, radius_km: float=50):

    target_coords = df[df['city'] == target_city][['latitude', 'longitude']].values[0]
    nearby_cities = df[df.apply(lambda x: great_circle(target_coords, (x['latitude'], x['longitude'])).km <= radius_km, axis=1)]['city'].unique()

    return nearby_cities


def calc_features(df: pd.DataFrame):

    # Create rolling statistics features
    for col in [c for c in df.columns if c.startswith("metric_")]:
        df[f"{col}_rolling_mean"] = df.groupby("city")[col].rolling(window=3).mean().reset_index(drop=True)
        df[f"{col}_rolling_std"] = df.groupby("city")[col].rolling(window=3).std().reset_index(drop=True)
        df[f"{col}_pct_change"] = df[col].pct_change()

    # Calculate nearby city network anomaly influence
    df['feature_nearby_anomaly_score_1h'] = (
        df.apply(lambda x:
                 df[(df['timestamp'] <= x['timestamp']) &
                    (df['timestamp'] > x['timestamp'] - pd.Timedelta(hours=1)) &
                    (df['city'].isin(find_nearby_cities(df, x['city'], radius_km=50)))][
                     [c for c in df.columns if c.startswith("metric_")]].std().mean(),
                 axis=1))
    df['feature_nearby_anomaly_score_15m'] = (
        df.apply(lambda x:
                 df[(df['timestamp'] <= x['timestamp']) &
                    (df['timestamp'] > x['timestamp'] - pd.Timedelta(minutes=15)) &
                    (df['city'].isin(find_nearby_cities(df, x['city'], radius_km=50)))][
                     [c for c in df.columns if c.startswith("metric_")]].std().mean(),
                 axis=1))

    df.dropna(inplace=True)


def model_anomalies(df: pd.DataFrame):

    features = [c for c in df.columns if c.startswith("metric_") or c.startswith("feature_")]

    # 1. Isolation Forest anomaly detection
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['iso_anomaly_score'] = iso_forest.fit_predict(df[features])
    df['iso_anomaly_score'] = df['iso_anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

    # 2. GMM for outage probability
    print("Training GMM...")
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)
    gmm = GaussianMixture(n_components=2, random_state=42)
    df['gmm_score'] = gmm.fit_predict(df_imputed)
    df['outage_probability'] = gmm.predict_proba(df_imputed)[:, 1]

    # 3. Variational Autoencoder for outage likelihood
    print("Training VAE...")
    input_dim = df[features].shape[1]
    input_layer = Input(shape=(input_dim,))

    encoder = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(2)  # Latent space representation
    ])

    decoder = models.Sequential([
        layers.InputLayer(input_shape=(2,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(input_dim)
    ])

    encoded = encoder(input_layer)
    decoded = decoder(encoded)
    vae = models.Model(inputs=input_layer, outputs=decoded)

    vae.compile(optimizer='adam', loss='mse')
    vae.fit(df[features], df[features], epochs=10, batch_size=32)

    df['vae_reconstruction_error'] = ((df[features] - vae.predict(df[features])) ** 2).mean(axis=1)
    df['outage_likelihood'] = (df['vae_reconstruction_error'] - df['vae_reconstruction_error'].min()) / (
                df['vae_reconstruction_error'].max() - df['vae_reconstruction_error'].min())

    # Visualization:
    plt.figure(figsize=(12, 5))
    df_mini = df[df["city"] == "Valencia"]
    plt.plot(df_mini['timestamp'], df_mini['outage_likelihood'], label="VAE")
    plt.plot(df_mini['timestamp'], df_mini['iso_anomaly_score'], label="Isolation Forest")
    plt.plot(df_mini['timestamp'], df_mini['outage_probability'], label="GMM")
    plt.axhline(y=0.8, color='r', linestyle='--', label="High Risk Threshold")
    plt.xlabel("Time")
    plt.ylabel("Outage Likelihood Score")
    plt.title("Network Outage Likelihood Over Time")
    plt.legend()
    plt.show(block=True)
    breakpoint()


# Define valid codes:
endpoint = "entities/query"
response = requests.get(API_URL + endpoint)
if response.status_code == 200:
    metadata = response.json()["data"]
    """
    [{
      "code": "AF",
      "name": "Africa",
      "type": "continent",
      "subnames": [],
      "attrs": {
        "fqid": "geo.netacuity.AF"
      },
    }, ...]
    """

# Define date range, region, desired signal
from_date_posix = str(int(datetime(year=2024, month=1, day=1).timestamp()))
to_date_posix = str(int(datetime(year=2025, month=1, day=1).timestamp()))
data_src = ""  # ("bgp", "merit-nt", "gtr", "gtr-norm", "ping-slash24")

country = "Spain"
region_codes = [
    item["code"] for item in metadata if
    item["type"] == "region" and
    item["attrs"]["country_name"] == country and
    "Invalid" not in item["name"]
]
entity_type = "region"  # ("continent", "country", "region")
entity_code = ",".join(region_codes)  # Comma-separated

# Define parameters to fetch global outages (modify as needed)
params = {
    "from": from_date_posix,
    "until": to_date_posix,
    "datasource": data_src,
}

# Fetch data from API
endpoint = f"signals/raw/{entity_type}/{entity_code}"
response = requests.get(API_URL + endpoint, params=params)
if response.status_code == 200:
    data = response.json()["data"]
    print("Data successfully fetched from IODA API")
else:
    print(f"Error fetching data: {response.status_code}")
    exit()


""" Example: BCN """

region = "Valencia"
# example_data = [item for item in data if item[0]["entityName"] == region][0]
df = merge_sources_into_df(data)

# Display basic statistics
print("\nBasic Data Overview:")
print(df[[c for c in df.columns if c.startswith("metric_")]].info())
print("\nSummary Statistics:")
print(df[[c for c in df.columns if c.startswith("metric_")]].describe())

f1, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 5), constrained_layout=True)
for n_metric, metric in enumerate([c for c in df.columns if c.startswith("metric")]):

    dataset = df[df["city"] == region][metric]
    metric_name = metric.split("_")[1]

    # Time-series plot of network activity
    ax = axs[0, n_metric]
    sns.lineplot(data=df, x="timestamp", y=dataset, ax=ax)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Value")
    ax.set_title(metric_name)
    ax.tick_params(axis="x", labelrotation=45)

    # Histogram of activity levels
    ax = axs[1, n_metric]
    sns.histplot(dataset, bins=30, kde=True, ax=ax)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

plt.suptitle(region)
plt.show(block=True)
breakpoint()

model_anomalies(df)
breakpoint()