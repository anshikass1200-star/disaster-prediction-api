from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import uvicorn

model = joblib.load("model/disaster_model.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    Disaster_Type: str
    Country: str
    Region: str
    Magnitude: float
    Sum_of_Latitude: float
    Sum_of_Longitude: float


@app.post("/predict")
def predict(data: PredictionRequest):

    input_dict = {
        "Disaster Type": data.Disaster_Type,
        "Country": data.Country,
        "Region": data.Region,
        "Magnitude": data.Magnitude,
        "Sum of Latitude": data.Sum_of_Latitude,
        "Sum of Longitude": data.Sum_of_Longitude,
    }

    df = pd.DataFrame([input_dict])

    df = df[[
        "Disaster Type",
        "Country",
        "Region",
        "Magnitude",
        "Sum of Latitude",
        "Sum of Longitude"
    ]]

    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    labels = {
        0: "Low Risk",
        1: "Flood",
        2: "Earthquake",
        3: "Cyclone"
    }

    return {
        "prediction": labels.get(int(prediction), "Unknown"),
        "confidence": float(max(probabilities))
    }


@app.get("/")
def home():
    return {"message": "Disaster Prediction API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)