from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================================================
# FRAUD DETECTION BACKEND
# React frontend -> FastAPI backend -> Hybrid ML models
# =========================================================


app = FastAPI(title="Fraud Detection Backend")


# React localhost bağlantısı için CORS ayarı
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent

FRAUD1_MODEL_PATH = BASE_DIR / "models" / "hybrid_model_fraud1.pkl"
FRAUD2_MODEL_PATH = BASE_DIR / "models" / "hybrid_model_fraud2.pkl"


# =========================================================
# REQUEST MODELS
# =========================================================


class RatioItem(BaseModel):
    code: str
    value: float | int | str | None = 0


class PredictionRequest(BaseModel):
    fraudType: str
    ratios: list[RatioItem]


# =========================================================
# MODEL LOADING
# =========================================================


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    return joblib.load(model_path)


fraud1_model = load_model(FRAUD1_MODEL_PATH)
fraud2_model = load_model(FRAUD2_MODEL_PATH)


# =========================================================
# RATIO CODE MAPPING
#
# Frontend'de görünen rasyo kodları ile modelin eğitim
# veri setindeki kolon adları aynı olmayabiliyor.
# Bu yüzden burada eşleştirme yapıyoruz.
# =========================================================


FRAUD1_RATIO_MAPPING = {
    "TB / Ö": "TB / Ö",
    "NFB / Ö": "NFB / Ö",
    "KVB / Ö": "KVB / Ö",
    "Cari Oran": "Cari Oran",
    "Asit Test Oranı": "Asit Test Oran",
    "N / KVB": "N / KVB",
    "NDK / Ö": "NDK / Ö",
    "İTTA / TTA": "İTTA / TTA",
    "İTTB / TTB": "İTTB / TTB",
    "MDV / Ö": "MDV / Ö",
    "H / TA": "H / TA",
    "H / MDV": "H / MDV",
    "TK / Ö": "TK / Ö",
    "FAVÖK / İDAFG": "FAVÖK / İDAFG",
    "FAVÖK Marjı": "FAVÖK Marjı",
    "BK / H": "BK / H",
    "VÖK / H": "VÖK / H",
    "NDK / TK": "NDK / TK",
    "VÖK / Fgid": "VÖK / Fgid",
    "SM / Stok": "SM / Stok",
    "H / TK": "H / TV",
    "H / Ö": "H / Ö",
}


FRAUD2_RATIO_MAPPING = {
    "TY / TÖ": "TY/TÖ",
    "NFB / TÖ": "NFB/TÖ",
    "TKVY / TÖ": "TKVY/TÖ",
    "T.DÖN.V / TKVY": "T.DÖN.V/ TKVY",
    "T.DÖN.V-Stok / TKVY": "T.DÖN.V-Stok/ TKVY",
    "NAKİT / TKVY": "NAKİT/ TKVY",
    "DK(Z) / TÖ": "DK(Z)/TÖ",
    "İTTA / TA": "İTTA/TA",
    "İTTB / TB": "İTTB/TB",
    "MDV / TÖ": "MDV/TÖ",
    "H / TA": "H/TA",
    "H / MDV": "H/MDV",
    "SFVÖK-EFK / MDV": "SFVÖK-EFK/MDV",
    "SFVÖK-EFK / TV": "SFVÖK-EFK/TV",
    "DK(Z) / MDV": "DK(Z)/MDV",
    "EFK / H": "EFK/H",
    "EFDG / H": "EFDG/H",
    "SFVÖK / EFK": "SFVÖK/EFK",
    "EFK / TV": "EFK/TV",
    "DK(Z) / TK": "DK(Z)/TK",
    "NetDönKar* / MDV*": "NetDönKar*/MDV*",
    "SFDK(Z) / TÖ": "SFDK(Z)/TÖ",
    "FAVÖK / T.DuranV": "FAVÖK/T.DuranV",
}


# =========================================================
# HELPERS
# =========================================================


def to_float(value):
    try:
        if value is None or value == "":
            return 0.0

        return float(value)

    except Exception:
        return 0.0


def build_model_input(ratios, selected_model, ratio_mapping):
    """
    Frontend'den gelen ratio listesi:
    [
      {"code": "TB / Ö", "value": 1.23},
      ...
    ]

    Modelin beklediği kolon sırasına göre tek satırlık DataFrame üretir.
    Eksik kolon varsa 0.0 basılır.
    """

    incoming_values = {}

    for item in ratios:
        frontend_code = item.code
        model_column = ratio_mapping.get(frontend_code, frontend_code)
        incoming_values[model_column] = to_float(item.value)

    row = {}

    for column in selected_model.feature_columns:
        row[column] = incoming_values.get(column, 0.0)

    return pd.DataFrame([row], columns=selected_model.feature_columns)


def get_positive_probability(model, probabilities):
    classes = np.array(model.classes)

    if 1 in classes:
        positive_index = int(np.where(classes == 1)[0][0])
    else:
        positive_index = int(np.argmax(classes))

    return float(probabilities[0][positive_index])


def build_prediction_response(model, input_df, fraud_type):
    probabilities = model.predict_proba(input_df)
    prediction = int(model.predict(input_df)[0])

    positive_probability = get_positive_probability(model, probabilities)

    if prediction == 1:
        prediction_label = "Hileli"
        risk_level = "Yüksek Risk"
    else:
        prediction_label = "Hilesiz"
        risk_level = "Düşük Risk"

    confidence_score = max(float(np.max(probabilities[0])), 0.0)

    return {
        "status": "success",
        "backendConnected": True,
        "fraudType": fraud_type,
        "prediction": prediction,
        "predictionLabel": prediction_label,
        "riskLevel": risk_level,
        "fraudProbability": round(positive_probability, 4),
        "confidenceScore": round(confidence_score, 4),
        "metrics": model.metrics,
        "confusionMatrix": model.confusion_matrix,
        "modelFeatureColumns": model.feature_columns,
        "message": "Model tahmini başarıyla üretildi.",
    }


# =========================================================
# ROUTES
# =========================================================


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Fraud Detection Backend çalışıyor.",
        "models": {
            "fraud1": FRAUD1_MODEL_PATH.exists(),
            "fraud2": FRAUD2_MODEL_PATH.exists(),
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "running",
        "fraud1_model_loaded": fraud1_model is not None,
        "fraud2_model_loaded": fraud2_model is not None,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    fraud_type = request.fraudType

    if fraud_type == "fraud1":
        input_df = build_model_input(
            ratios=request.ratios,
            selected_model=fraud1_model,
            ratio_mapping=FRAUD1_RATIO_MAPPING,
        )

        return build_prediction_response(
            model=fraud1_model,
            input_df=input_df,
            fraud_type=fraud_type,
        )

    if fraud_type == "fraud2":
        input_df = build_model_input(
            ratios=request.ratios,
            selected_model=fraud2_model,
            ratio_mapping=FRAUD2_RATIO_MAPPING,
        )

        return build_prediction_response(
            model=fraud2_model,
            input_df=input_df,
            fraud_type=fraud_type,
        )

    return {
        "status": "error",
        "backendConnected": True,
        "message": f"Geçersiz fraudType değeri: {fraud_type}",
    }