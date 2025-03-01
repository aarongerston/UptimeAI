"""
main.py

Orchestrates the entire data ingestion, processing, and modeling pipeline.
"""

import logging
import sys
from datetime import datetime

import pandas as pd

from config import (
    DEFAULT_FROM_DATE,
    DEFAULT_TO_DATE,
    DEFAULT_DATASOURCE,
    DEFAULT_COUNTRY,
    logger,
)
from data_fetch import get_available_entities, fetch_signal_data
from data_processing import merge_sources_into_df
from feature_engineering import calc_features
from modeling import fit_isolation_forest, fit_gmm, fit_vae, plot_anomaly_results


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

    # 2) Filter region codes for given country
    region_codes = []
    for item in metadata:
        # item["attrs"] = {"country_name": "...", "fqid": "...", ...}
        if item.get("type") == "region":
            attrs = item.get("attrs", {})
            if (
                attrs.get("country_name") == DEFAULT_COUNTRY
                and "Invalid" not in item["name"]
            ):
                region_codes.append(item["code"])

    if not region_codes:
        logger.error(f"No valid region codes found for {DEFAULT_COUNTRY}. Exiting.")
        sys.exit(1)

    entity_type = "region"  # Could be "country" or "continent"
    entity_code = ",".join(region_codes)  # Comma-separated

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
    df = calc_features(df, scale_output_path="scaler.pkl")

    # 6) Run anomaly detection (IsolationForest, GMM, VAE)
    feature_cols = [
        c for c in df.columns if c.startswith("metric_") or c.startswith("feature_")
    ]

    df = fit_isolation_forest(
        df, features=feature_cols, contamination=0.05, iso_model_path="iso_forest.pkl"
    )
    df = fit_gmm(df, features=feature_cols, gmm_model_path="gmm_model.pkl")
    df = fit_vae(df, features=feature_cols, vae_model_path="vae_model")

    # 7) Plot anomaly results for a chosen example city
    example_city = "Valencia"
    plot_anomaly_results(df, example_city, output_fig="valencia_anomalies.png")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
