from fastapi import FastAPI
from pydantic import BaseModel
from src.predict_jit import predict_url

app = FastAPI(title="PhishDetect API")

class InputData(BaseModel):
    text: str


@app.post("/predict")
def predict(data: InputData):
    label, legit_prob, phish_prob = predict_url(data.text)

    return {
        "label": label,
        "legitimate_probability": legit_prob,
        "phishing_probability": phish_prob,
    }
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
