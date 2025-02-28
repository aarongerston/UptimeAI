
import os
import uvicorn
from typing import Optional
from fastapi import FastAPI, Query

from aux import fetch_ioda_data, ioda2df, predict_outages

# Initialize FastAPI app
app = FastAPI()


@app.get("/predict")
def predict_api(
        country: Optional[str] = Query(None, description="Filter regions by continent"),
        continent: Optional[str] = Query(None, description="Filter regions by country"),
        region: Optional[str] = Query(None, description="Get data from a specific region"),
):

    # Fetch data from IODA API
    data = fetch_ioda_data(continent=continent, country=country, region=region)
    if not any(data):
        return {"error": "No data available from IODA"}

    # Structure data as DataFrame
    df = ioda2df(data)

    # Apply the black-box prediction function
    predictions = predict_outages(df)

    return {"predictions": predictions}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)