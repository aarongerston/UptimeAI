"""
train_models.py

Orchestrates the entire data ingestion, processing, and modeling pipeline.
"""
import os
import sys

from functions.config import (
    DEFAULT_FROM_DATE,
    DEFAULT_TO_DATE,
    DEFAULT_DATASOURCE,
    DEFAULT_COUNTRY,
    DEFAULT_CONTINENT,
    logger,
)
from functions.data_fetch import get_available_entities, fetch_signal_data
from functions.data_processing import merge_sources_into_df
from functions.feature_engineering import calc_features
from functions.modeling import fit_isolation_forest, fit_gmm, fit_vae, plot_anomaly_results

import matplotlib
matplotlib.use("TkAgg")


def main():
    """
    End-to-end pipeline that:
    1) Fetches metadata
    2) Determines region codes for the given country
    3) Fetches raw signal data
    4) Merges into a DataFrame with geocoding
    5) Computes features
    6) Runs anomaly detection models (IForest, GMM, VAE)
    7) Plots anomaly results for a chosen example city
    """
    logger.info("Starting the ML pipeline...")

    # 1) Fetch entity metadata
    metadata = get_available_entities()
    if not metadata:
        logger.error("No metadata returned. Exiting.")
        sys.exit(1)

    continent_codes = {item["name"]: item["code"] for item in metadata if item["type"] == "continent"}
    country_codes = {item["name"]: item["code"] for item in metadata if item["type"] == "country"}
    region_codes = {item["name"]: item["code"] for item in metadata if item["type"] == "region"}

    # 2) Filter region codes for given country or continent
    filtered_codes = []
    country_code = DEFAULT_COUNTRY if DEFAULT_COUNTRY in country_codes.values() \
        else country_codes[DEFAULT_COUNTRY] if DEFAULT_COUNTRY in country_codes.keys() \
        else None
    continent_code = DEFAULT_CONTINENT if DEFAULT_CONTINENT in continent_codes.values() \
        else continent_codes[DEFAULT_CONTINENT] if DEFAULT_CONTINENT in continent_codes.keys() \
        else None
    for item in metadata:
        # item["attrs"] = {"country_name": "...", "fqid": "...", ...}
        if item.get("type") == "region":
            if (
                ((item["attrs"]["fqid"].split(".")[3] == country_code) if isinstance(country_code, str) else True) and
                ((item["attrs"]["fqid"].split(".")[2] == continent_code) if isinstance(continent_code, str) else True)
                and "Invalid" not in item["name"]
            ):
                filtered_codes.append(item["code"])

    if not filtered_codes:
        logger.error(f"No valid region codes found for {DEFAULT_COUNTRY}. Exiting.")
        sys.exit(1)

    entity_type = "region"  # Could be "country" or "continent"
    entity_code = ",".join(filtered_codes)  # Comma-separated

    # 3) Fetch raw signal data
    data = fetch_signal_data(
        entity_type=entity_type,
        entity_code=entity_code,
        from_date=DEFAULT_FROM_DATE,
        to_date=DEFAULT_TO_DATE,
        datasource=DEFAULT_DATASOURCE,
    )

    if not data:
        logger.error("No data returned from API. Exiting.")
        sys.exit(1)

    # 4) Merge data into a DataFrame
    df = merge_sources_into_df(data)
    if df.empty:
        logger.error("Merged DataFrame is empty after geocoding. Exiting.")
        sys.exit(1)

    logger.info(f"Merged DataFrame shape: {df.shape}")

    # 5) Compute features
    df = calc_features(df, metric_scaler_path="backend/metric_scaler.pkl", feature_scaler_path="backend/feature_scaler.pkl")

    # 6) Run anomaly detection (IsolationForest, GMM, VAE)
    feature_cols = [
        c for c in df.columns if c.startswith("metric_") or c.startswith("feature_")
    ]

    df = fit_isolation_forest(
        df, features=feature_cols, contamination=0.05, iso_model_path="iso_forest.pkl"
    )
    df = fit_gmm(df, features=feature_cols, gmm_model_path="gmm_model.pkl")
    df = fit_vae(df, features=feature_cols, vae_model_path="vae_model.keras")

    # 7) Plot anomaly results for a chosen example city
    os.makedirs(f"figures/{continent_code}", exist_ok=True)
    for city in df["city"].unique():
        plot_anomaly_results(df, city, output_fig=f"figures/{continent_code}/{city}_anomalies.png")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
