"""
feature_engineering.py

Contains functions to create and transform feature sets
(e.g., scaling, rolling stats, lagged features, and
aggregations based on nearby cities).
"""

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import joblib

from data_processing import find_nearby_cities

logger = logging.getLogger(__name__)


def calc_features(
    df: pd.DataFrame, scale_output_path: str = "scaler.pkl"
) -> pd.DataFrame:
    """
    Calculate rolling statistics, lagged features, and scale metrics in-place.
    Also compute aggregated features from nearby cities.

    :param df: Input DataFrame with columns
               [timestamp, city, country_code, metric_*, latitude, longitude]
    :param scale_output_path: Path to save the fitted scaler
    :return: DataFrame with new feature columns.
    """
    # Identify metric columns
    metrics = [c for c in df.columns if c.startswith("metric_")]
    if not metrics:
        logger.warning(
            "No metric columns found to scale. Returning original DataFrame."
        )
        return df

    # 1. Scale metrics with RobustScaler
    logger.info("Scaling raw metric columns with RobustScaler.")
    scaler = RobustScaler()
    df[metrics] = scaler.fit_transform(df[metrics])
    joblib.dump(scaler, scale_output_path)

    # 2. Calculate rolling MAD for each metric
    n_timesteps = 3
    for metric in metrics:
        col_name = f"calc_{metric.split('_', 1)[1]}_mad"
        df[col_name] = (
            df.groupby("city")[metric]
            .rolling(window=n_timesteps, min_periods=2)
            .apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
            .reset_index(level=0, drop=True)
        )

    # 3. Calculate an overall MAD score
    mad_cols = [c for c in df.columns if "calc_" in c and "_mad" in c]
    if mad_cols:
        df["calc_avg_mad"] = df[mad_cols].mean(axis=1)
    else:
        df["calc_avg_mad"] = np.nan

    # 4. Create lagged features for rolling window size ~ 15 min
    feature_cols = [c for c in df.columns if c.startswith("calc_")]
    if "timestamp" not in df.columns:
        logger.warning(
            "No timestamp column found for lagging. Returning partial DataFrame."
        )
        return df

    # Approximate median sampling interval (in seconds) across entire dataset
    df["time_diff"] = df.groupby("city")["timestamp"].diff().dt.total_seconds()
    time_step = (
        int(np.nanmedian(df["time_diff"].dropna()))
        if not df["time_diff"].dropna().empty
        else 60
    )
    n_lags = max(1, 15 // (time_step // 60))  # 15 minutes / (time_step in minutes)

    logger.info(
        f"Median sampling interval: ~{time_step} seconds. Creating {n_lags} lag(s)."
    )
    for lag in range(1, n_lags + 1):
        for col in feature_cols:
            df[f"{col}_lag_{int(lag * time_step // 60)}m"] = df.groupby("city")[
                col
            ].shift(lag)

    # 5. Aggregate recent data from nearby cities
    logger.info("Building dictionary of nearby cities for up to 150km radius.")
    df_clean = df.dropna(subset=["latitude", "longitude"])
    city_list = df_clean["city"].unique()
    nearby_cities_dict = {
        city: find_nearby_cities(df_clean, city, radius_km=150)
        for city in tqdm(city_list)
    }

    # Create aggregated features for each timestamp from neighbors
    lag_cols = [c for c in df.columns if "_lag_" in c]
    if not lag_cols:
        logger.warning(
            "No lagged columns found. Returning DataFrame without neighbor aggregation."
        )
        return df

    df_nearby_list = []
    for city in city_list:
        neighbors = nearby_cities_dict.get(city, [])
        if len(neighbors) == 0:
            continue
        df_subset = df[df["city"].isin(neighbors)]
        # Mean of lagged features among neighbors
        df_agg = df_subset.groupby("timestamp", as_index=False)[lag_cols].mean()
        # Rename columns
        rename_map = {col: f"feature_{col}_nearby" for col in lag_cols}
        df_agg.rename(columns=rename_map, inplace=True)
        # Assign city to the aggregated data
        df_agg["city"] = city
        df_nearby_list.append(df_agg)

    if df_nearby_list:
        df_nearby = pd.concat(df_nearby_list, ignore_index=True)
        # Merge back
        df = pd.merge(df, df_nearby, on=["city", "timestamp"], how="left")

    df.drop(columns=["time_diff"], inplace=True, errors="ignore")
    return df
