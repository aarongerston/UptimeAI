
import joblib
import warnings
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from warnings import warn
import matplotlib.pyplot as plt
from geopy.distance import great_circle
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import layers, models, Input

import matplotlib
matplotlib.use("TkAgg")
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    df.sort_values(by=["city", "timestamp"], inplace=True)
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

    df.dropna(subset = ['longitude', 'latitude'], inplace=True)

    return df


def find_nearby_cities(df: pd.DataFrame, target_city: str, radius_km: float=100):

    # Ensure the target city exists in the cleaned DataFrame
    target_row = df[df['city'] == target_city]

    if target_row.empty:
        print(f"Warning: No valid coordinates found for {target_city}. Returning empty list.")
        return []

    # Extract target coordinates
    target_coords = target_row[['latitude', 'longitude']].values[0]

    # Compute distances only for valid locations
    nearby_cities = df[df.apply(
        lambda x: great_circle(target_coords, (x['latitude'], x['longitude'])).km <= radius_km, axis=1
    )]['city'].unique()

    return nearby_cities


def calc_features(df: pd.DataFrame):

    # Scale metrics:
    metrics = [c for c in df.columns if c.startswith("metric_")]
    scaler = RobustScaler()  # scales using median and IQR
    df[metrics] = scaler.fit_transform(df[metrics])
    joblib.dump(scaler, "scaler.pkl")  # Stores scaler as a .pkl file to be used during inference

    # Calculate MAD over last 3 values for each metric
    n_timesteps = 3
    for metric in metrics:
        df[f"calc_{metric.split('_')[1]}_mad"] = df[metric].rolling(window=n_timesteps, min_periods=2).apply(
            lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True
        )

    # Calculate an overall MAD score:
    mad_cols = [c for c in df.columns if "calc_" in c and "_mad" in c]
    df["calc_avg_mad"] = df[mad_cols].mean(axis=1)

    # # Create lagged features (previous 15 mins)
    feature_cols = [col for col in df.columns if col.startswith("calc_")]
    max_lag = 15  # minutes
    time_diffs = df["timestamp"].diff().dt.total_seconds().dropna()
    time_step = int(np.nanmedian(time_diffs))  # seconds
    n_lags = max(1, max_lag // (time_step // 60))  # Ensure at least 1 lag is used
    for lag in range(1, n_lags + 1):  # Lags 1, 2, 3
        for col in feature_cols:
            feature = col.split("_")[1]
            df[f"{feature}_lag_{int(lag * time_step // 60)}m"] = df.groupby("city")[col].shift(lag)

    # Calculate recent overall variability in nearby areas
    df_clean = df.dropna(subset=['latitude', 'longitude']).copy()
    iterator = tqdm(df['city'].unique(), desc="Finding nearby cities", leave=False)
    nearby_cities_dict = {city: find_nearby_cities(df_clean, city, radius_km=150) for city in iterator}
    # Create a new DataFrame to store nearby city aggregated features
    lag_cols = [c for c in df.columns if "_lag_" in c]
    df_nearby_list = []
    for city in df["city"].unique():
        nearby_cities = nearby_cities_dict.get(city, [])
        if not any(nearby_cities):
            continue  # Skip if there are no nearby cities
        # Filter df for only nearby cities
        df_nearby_subset = df[df["city"].isin(nearby_cities)]
        # Compute the mean of lagged features for nearby cities per timestamp
        df_nearby_agg = df_nearby_subset.groupby("timestamp", as_index=False)[lag_cols].mean()
        df_nearby_agg.rename(columns={col: f"feature_{col}_nearby" for col in lag_cols}, inplace=True)
        # Assign the computed values to the original city
        df_nearby_agg["city"] = city  # Keeps "city" column intact
        df_nearby_list.append(df_nearby_agg)

    # Concatenate all computed nearby city data
    df_nearby = pd.concat(df_nearby_list, ignore_index=True)
    # Merge back with original df without duplicate city columns
    df = df.merge(df_nearby, on=["city", "timestamp"], how="left")

    return df


def model_anomalies(df: pd.DataFrame):

    features = [c for c in df.columns if c.startswith("metric_") or c.startswith("feature_")]

    # Impute missing data
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

    # 1. Isolation Forest anomaly detection
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['iso_anomaly_score'] = iso_forest.fit_predict(X)
    df['iso_anomaly_score'] = df['iso_anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

    # 2. GMM for outage probability
    print("Training GMM...")

    # Test different numbers of components
    bic_scores = []
    aic_scores = []
    components_range = list(range(1, X.shape[1]))  # Try from 1 to n_features components
    for n in components_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)

        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    plt.figure(figsize=(10, 5))
    plt.ion()
    plt.plot(components_range, bic_scores, label="BIC", marker='o')
    plt.plot(components_range, aic_scores, label="AIC", marker='s')
    plt.xlabel("Number of Components")
    plt.ylabel("BIC / AIC Score")
    plt.legend()
    plt.title("Choosing the Optimal Number of Components in GMM")
    plt.show(block=False)

    # Locate the elbow
    # try:
    #     n_components = components_range[np.argwhere(np.diff(np.diff(bic_scores)) < 0)[0][0]]
    # except IndexError:
    n_components = components_range[np.argmax(np.diff(np.diff(bic_scores))) + 1]

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    df['gmm_score'] = gmm.fit_predict(X)
    df["gmm_prediction"] = df["gmm_score"] != 0
    # df['outage_probability'] = gmm.predict_proba(X)[:, 1]

    # 3. Variational Autoencoder for outage likelihood
    print("Training VAE...")
    input_dim = X.shape[1]
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
    vae.fit(X, X, epochs=10, batch_size=32)

    df['vae_reconstruction_error'] = ((X - vae.predict(X)) ** 2).mean(axis=1)

    # Remove outliers
    lower_bound = df['vae_reconstruction_error'].quantile(1 / 100)
    upper_bound = df['vae_reconstruction_error'].quantile(99 / 100)
    df['vae_reconstruction_error_clipped'] = df['vae_reconstruction_error'].clip(lower_bound, upper_bound)

    # Apply Min-Max Scaling after clipping
    df['outage_likelihood'] = (df['vae_reconstruction_error_clipped'] - df['vae_reconstruction_error_clipped'].min()) / (
                                df['vae_reconstruction_error_clipped'].max() - df['vae_reconstruction_error_clipped'].min())

    # Visualization:
    plt.figure(figsize=(12, 5))
    plt.ion()
    df_mini = df[df["city"] == "Valencia"]
    plt.plot(df_mini['timestamp'], df_mini['outage_likelihood'], label="VAE")
    plt.plot(df_mini['timestamp'], df_mini['iso_anomaly_score'], label="Isolation Forest")
    plt.plot(df_mini['timestamp'], df_mini['gmm_prediction'], label="GMM")
    plt.axhline(y=0.8, color='r', linestyle='--', label="High Risk Threshold")
    plt.xlabel("Time")
    plt.ylabel("Outage Likelihood Score")
    plt.title("Network Outage Likelihood Over Time")
    plt.legend()
    plt.show(block=False)
    breakpoint()


def plot_features(df: pd.DataFrame, features: list[str], region: str):

    f1, axs = plt.subplots(nrows=2, ncols=len(features), figsize=(12, 5), constrained_layout=True)
    for n_metric, metric in enumerate(features):
        dataset = df[df["city"] == region][metric]
        metric_name = " ".join(metric.split("_")[1:])

        # Time-series plot of network activity
        ax = axs[0, n_metric]
        sns.lineplot(data=df, x="timestamp", y=dataset, ax=ax)
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Value") if n_metric == 0 else None
        ax.set_title(metric_name)
        ax.tick_params(axis="x", labelrotation=45)

        # Histogram of activity levels
        ax = axs[1, n_metric]
        sns.histplot(dataset, bins=30, kde=True, ax=ax)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    [ax.sharex(axs[0, 0]) for ax in axs[0]]  # top row shares x-axes
    plt.suptitle(region)
    plt.show(block=False)


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
from_date_posix = str(int(datetime(year=2024, month=12, day=31).timestamp()))
to_date_posix = str(int(datetime(year=2025, month=1, day=2).timestamp()))
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

# example_data = [item for item in data if item[0]["entityName"] == region][0]
df = merge_sources_into_df(data)

df = calc_features(df)

features = [c for c in df.columns if c.startswith(("metric_", "feature_"))]
plt.ion()
region = "Valencia"
plot_features(df, features=[c for c in df.columns if c.startswith("metric_")], region=region)
plot_features(df, features=[c for c in df.columns if c.startswith("feature_")], region=region)
breakpoint()

model_anomalies(df)
breakpoint()