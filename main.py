import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load the trained model
with open("congestion_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Define the expected input format
class CongestionInput(BaseModel):
    is_holiday: int
    is_weekend: int
    Friday: int
    Monday: int
    Saturday: int
    Sunday: int
    Thursday: int
    Tuesday: int
    Wednesday: int
    Cloudy: int
    Rainy: int
    Sunny: int
    Windy: int
    _8_00_8_30: int
    _8_30_9_00: int
    _9_00_9_30: int
    _9_30_10_00: int
    _10_00_10_30: int
    _10_30_11_00: int
    _11_00_11_30: int
    _11_30_12_00: int
    _12_00_12_30: int
    _12_30_13_00: int
    _13_00_13_30: int
    _13_30_14_00: int
    _14_00_14_30: int
    _14_30_15_00: int
    _15_00_15_30: int
    _15_30_16_00: int
    _16_00_16_30: int
    _16_30_17_00: int
    _17_00_17_30: int
    _17_30_18_00: int
    _18_00_18_30: int
    _18_30_19_00: int
    _19_00_19_30: int
    _19_30_20_00: int
@app.get("/")
def home():
    return {"message": "Congestion Prediction API running!"}
@app.post("/predict")
def predict(data: CongestionInput):
    input_array = np.array([list(data.dict().values())])
    prediction = model.predict(input_array)[0]
    return {"predicted_congestion": prediction}

# Run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
