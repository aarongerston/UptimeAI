
import os
import uvicorn
from typing import Optional
from fastapi import FastAPI, Query

from backend.aux import fetch_ioda_data, ioda2df, predict_outages
from functions.feature_engineering import calc_features

# Initialize FastAPI app
app = FastAPI()


@app.get("/predict")
def predict_api(
        country: Optional[str] = Query(None, description="Filter regions by country"),
        continent: Optional[str] = Query(None, description="Filter regions by continent"),
        region: Optional[str] = Query(None, description="Get data from a specific region"),
):

    # Fetch data from IODA API
    data = fetch_ioda_data(continent=continent, country=country, region=region)
    if not any(data):
        return {"error": "No data available from IODA"}

    # Structure data as DataFrame
    df = ioda2df(data)

    # Calculate features
    df = calc_features(df, metric_scaler_path="backend/metric_scaler.pkl", feature_scaler_path="backend/feature_scaler.pkl")

    # Apply the black-box prediction function
    predictions = predict_outages(df, model_path="backend/vae_model.keras")

    return {"predictions": predictions}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)