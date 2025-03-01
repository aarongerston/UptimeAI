"""
modeling.py

Implements various anomaly detection algorithms
(Isolation Forest, GMM, VAE) on the feature DataFrame.
Includes functions for plotting results.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)


def fit_isolation_forest(
    df: pd.DataFrame,
    features: list[str],
    contamination: float = 0.05,
    iso_model_path: str = "iso_forest.pkl",
) -> pd.DataFrame:
    """
    Fits an IsolationForest on the given feature columns and
    appends an anomaly flag to the DataFrame.

    :param df: DataFrame with feature columns
    :param features: List of feature column names to use
    :param contamination: Proportion of outliers
    :param iso_model_path: Path to save the trained model
    :return: DataFrame with a new 'iso_anomaly_flag' column.
    """
    logger.info("Fitting Isolation Forest...")
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_scores = iso_forest.fit_predict(X)
    df["iso_anomaly_flag"] = (anomaly_scores == -1).astype(int)

    joblib.dump(iso_forest, iso_model_path)
    return df


def fit_gmm(
    df: pd.DataFrame, features: list[str], gmm_model_path: str = "gmm_model.pkl"
) -> pd.DataFrame:
    """
    Fits a Gaussian Mixture Model (GMM) to the feature set
    and appends the cluster predictions to the DataFrame.

    :param df: DataFrame with feature columns
    :param features: List of feature column names
    :param gmm_model_path: Path to save the trained model
    :return: DataFrame with new columns 'gmm_cluster' and 'gmm_is_anomalous'
             based on cluster membership.
    """
    logger.info("Fitting GMM with BIC-based component selection...")

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

    # Evaluate BIC for range of components
    bic_scores = []
    aic_scores = []
    components_range = range(
        1, min(len(features), 10) + 1
    )  # limit max components for performance
    for n in components_range:
        gmm_temp = GaussianMixture(n_components=n, random_state=42)
        gmm_temp.fit(X)
        bic_scores.append(gmm_temp.bic(X))
        aic_scores.append(gmm_temp.aic(X))

    # Choose n_components based on the best BIC
    best_idx = np.argmin(bic_scores)
    best_n = list(components_range)[best_idx]
    logger.info(f"Selected {best_n} GMM components based on BIC scoring.")

    # Fit final GMM
    gmm = GaussianMixture(n_components=best_n, random_state=42)
    cluster_labels = gmm.fit_predict(X)
    df["gmm_cluster"] = cluster_labels

    # Optional: define cluster 0 as "normal" or use other logic
    # For a real pipeline, you'd identify which cluster(s) represent anomalies
    # Here, we do a simplistic approach: everything not in cluster 0 is "anomalous"
    df["gmm_is_anomalous"] = (df["gmm_cluster"] != 0).astype(int)

    joblib.dump(gmm, gmm_model_path)
    return df


def fit_vae(
    df: pd.DataFrame, features: list[str], vae_model_path: str = "vae_model"
) -> pd.DataFrame:
    """
    Fits a simple Variational Autoencoder for anomaly detection
    (reconstruction error). Appends 'outage_likelihood' column.

    :param df: DataFrame with feature columns
    :param features: List of feature column names
    :param vae_model_path: Path to save the trained VAE model
    :return: DataFrame with new columns for reconstruction error and likelihood.
    """
    logger.info("Fitting Variational Autoencoder (VAE)...")

    # Simple Imputer
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

    input_dim = X.shape[1]

    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    e = layers.Dense(16, activation="relu")(encoder_input)
    e = layers.Dense(8, activation="relu")(e)
    latent = layers.Dense(2, name="latent_layer")(e)

    # Decoder
    decoder_input = layers.Input(shape=(2,))
    d = layers.Dense(8, activation="relu")(decoder_input)
    d = layers.Dense(16, activation="relu")(d)
    decoder_output = layers.Dense(input_dim)(d)

    encoder = models.Model(encoder_input, latent, name="encoder")
    decoder = models.Model(decoder_input, decoder_output, name="decoder")

    vae_input = layers.Input(shape=(input_dim,))
    encoded = encoder(vae_input)
    reconstructed = decoder(encoded)
    vae = models.Model(vae_input, reconstructed, name="vae_model")
    vae.compile(optimizer="adam", loss="mse")

    # Fit VAE
    vae.fit(X, X, epochs=10, batch_size=32, verbose=0)
    vae.save(vae_model_path)

    # Calculate reconstruction error
    reconstructed_X = vae.predict(X)
    recons_error = np.mean((X - reconstructed_X) ** 2, axis=1)
    df["vae_reconstruction_error"] = recons_error

    # Clip the extremes (1%, 99%)
    lower_bound = np.percentile(recons_error, 1)
    upper_bound = np.percentile(recons_error, 99)
    clipped = np.clip(recons_error, lower_bound, upper_bound)

    # Min-Max scale
    df["outage_likelihood"] = (clipped - clipped.min()) / (
        clipped.max() - clipped.min()
    )

    return df


def plot_anomaly_results(
    df: pd.DataFrame, city_name: str, output_fig: str = "anomaly_plot.png"
) -> None:
    """
    Plot anomaly scores for a specific city over time and save to a file.

    :param df: DataFrame containing 'timestamp', 'city',
               'outage_likelihood', 'iso_anomaly_flag', 'gmm_is_anomalous'
    :param city_name: Name of the city to plot
    :param output_fig: Name of the file to save the plot
    """
    df_city = df[df["city"] == city_name].copy()
    if df_city.empty:
        logger.warning(f"No data found for city: {city_name}")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df_city["timestamp"], df_city["outage_likelihood"], label="VAE Likelihood")
    plt.plot(
        df_city["timestamp"], df_city["iso_anomaly_flag"], label="Isolation Forest Flag"
    )
    plt.plot(df_city["timestamp"], df_city["gmm_is_anomalous"], label="GMM Flag")
    plt.axhline(y=0.8, linestyle="--", label="High Risk Threshold")
    plt.xlabel("Time")
    plt.ylabel("Anomaly Score / Likelihood")
    plt.title(f"Anomaly Indicators for {city_name}")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_fig)
    plt.close()
    logger.info(f"Anomaly plot saved to {output_fig}.")
