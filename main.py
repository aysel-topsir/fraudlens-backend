from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================================================
# FRAUD DETECTION BACKEND
# React frontend -> FastAPI backend -> Hybrid ML models
# =========================================================


app = FastAPI(title="Fraud Detection Backend")


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
        "analysisFocus": instruction.strip() if instruction else "",
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


def build_upload_preview(df: pd.DataFrame):
    row_count = int(df.shape[0])
    column_count = int(df.shape[1])
    columns = list(df.columns)

    missing_values = df.isna().sum()
    total_missing = int(missing_values.sum())

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_column_count = len(numeric_columns)

    duplicate_rows = int(df.duplicated().sum())

    class_distribution = {}

    if "Class" in df.columns:
        class_counts = df["Class"].value_counts(dropna=False).to_dict()
        class_distribution = {
            str(key): int(value) for key, value in class_counts.items()
        }

    preview_rows = df.head(5).fillna("").to_dict(orient="records")

    return {
        "status": "success",
        "message": "CSV dosyası başarıyla analiz edildi.",
        "datasetOverview": {
            "rowCount": row_count,
            "columnCount": column_count,
            "numericColumnCount": numeric_column_count,
            "duplicateRows": duplicate_rows,
            "totalMissingValues": total_missing,
            "hasClassColumn": "Class" in df.columns,
        },
        "columns": columns,
        "classDistribution": class_distribution,
        "missingValues": {
            str(column): int(value)
            for column, value in missing_values.items()
            if int(value) > 0
        },
        "previewRows": preview_rows,
    }


def build_data_analyst_report(df: pd.DataFrame, instruction: str = ""):
    row_count = int(df.shape[0])
    column_count = int(df.shape[1])
    duplicate_rows = int(df.duplicated().sum())

    missing_values = df.isna().sum()
    total_missing = int(missing_values.sum())

    numeric_df = df.select_dtypes(include=[np.number])
    numeric_columns = numeric_df.columns.tolist()

    statistics_summary = {}

    for column in numeric_df.columns:
        statistics_summary[column] = {
            "mean": round(float(numeric_df[column].mean()), 4),
            "std": round(float(numeric_df[column].std()), 4),
            "min": round(float(numeric_df[column].min()), 4),
            "max": round(float(numeric_df[column].max()), 4),
        }

    class_distribution = {}

    if "Class" in df.columns:
        class_counts = df["Class"].value_counts(dropna=False).to_dict()
        class_distribution = {
            str(key): int(value) for key, value in class_counts.items()
        }

    outlier_summary = {}

    for column in numeric_df.columns:
        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(
            ((numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)).sum()
        )

        if outlier_count > 0:
            outlier_summary[column] = {
                "outlierCount": outlier_count,
                "lowerBound": round(float(lower_bound), 4),
                "upperBound": round(float(upper_bound), 4),
            }

    target_correlations = {}

    if "Class" in numeric_df.columns:
        correlations = (
            numeric_df.corr(numeric_only=True)["Class"]
            .drop(labels=["Class"], errors="ignore")
            .dropna()
            .sort_values(key=lambda values: values.abs(), ascending=False)
        )

        target_correlations = {
            str(column): round(float(value), 4)
            for column, value in correlations.head(10).items()
        }

    feature_redundancy = []

    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.drop(columns=["Class"], errors="ignore").corr().abs()

        for i, column_a in enumerate(corr_matrix.columns):
            for column_b in corr_matrix.columns[i + 1:]:
                corr_value = corr_matrix.loc[column_a, column_b]

                if corr_value >= 0.90:
                    feature_redundancy.append(
                        {
                            "featureA": str(column_a),
                            "featureB": str(column_b),
                            "correlation": round(float(corr_value), 4),
                        }
                    )

    focus_text = instruction.strip()
    focus_lower = focus_text.lower()

    focus_recommendations = []

    if focus_text:
        focus_recommendations.append(
            f"Analysis focus applied: {focus_text}. The report was prioritized according to this analyst-defined focus."
        )

    if any(word in focus_lower for word in ["outlier", "outliear", "aykırı", "aykiri"]):
        focus_recommendations.append(
            "Outlier-focused review: Variables with high IQR-based outlier counts should be prioritized before model training. Outlier capping may be considered, but the original dataset is not automatically modified."
        )

    if any(word in focus_lower for word in ["class", "sınıf", "sinif", "imbalance", "dengesiz", "denge", "balance", "class balance"]):
        if class_distribution:
            counts = list(class_distribution.values())
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = round(max_count / min_count, 2) if min_count else None
            focus_recommendations.append(
                f"Class-balance-focused review: The observed class distribution should be evaluated before modeling. Majority/minority ratio is approximately {imbalance_ratio}."
            )
        else:
            focus_recommendations.append(
                "Class-balance-focused review: No Class column was detected, so class imbalance could not be computed."
            )

    if any(word in focus_lower for word in ["missing", "eksik", "null", "nan"]):
        focus_recommendations.append(
            f"Missing-value-focused review: Total missing values detected: {total_missing}. Missingness should be handled before model training if present."
        )

    if any(word in focus_lower for word in ["correlation", "korelasyon", "redundancy"]):
        focus_recommendations.append(
            "Correlation-focused review: Highly correlated feature pairs should be examined for redundancy before feature selection and model training."
        )

    base_recommendations = [
        "Sınıf dağılımı modelleme öncesinde kontrol edilmelidir.",
        "Eksik değer bulunması durumunda uygun imputasyon stratejisi belirlenmelidir.",
        "Aykırı değerler IQR tabanlı sınırlandırma yöntemiyle değerlendirilebilir.",
        "Yüksek korelasyonlu değişkenler özellik tekrarları açısından incelenmelidir.",
    ]

    report = {
        "analysisFocus": focus_text,
        "datasetOverview": {
            "rowCount": row_count,
            "columnCount": column_count,
            "numericColumnCount": len(numeric_columns),
            "duplicateRows": duplicate_rows,
            "totalMissingValues": total_missing,
            "hasClassColumn": "Class" in df.columns,
        },
        "classDistribution": class_distribution,
        "missingValues": {
            str(column): int(value)
            for column, value in missing_values.items()
            if int(value) > 0
        },
        "statistics": statistics_summary,
        "outliers": outlier_summary,
        "targetCorrelations": target_correlations,
        "featureRedundancy": feature_redundancy,
        "recommendations": focus_recommendations + base_recommendations,
    }

    return {
        "status": "success",
        "message": "Data Analyst raporu başarıyla üretildi.",
        "analysisFocus": focus_text,
        "report": report,
    }


def build_feature_selector_report(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if "Class" not in numeric_df.columns:
        return {
            "status": "error",
            "message": "Feature Selector için Class kolonu gereklidir.",
        }

    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    y = numeric_df["Class"]
    X = numeric_df.drop(columns=["Class"], errors="ignore")

    if X.shape[1] == 0:
        return {
            "status": "error",
            "message": "Feature Selector için sayısal özellik bulunamadı.",
        }

    k_features = min(14, X.shape[1])

    target_correlations = (
        numeric_df.corr(numeric_only=True)["Class"]
        .drop(labels=["Class"], errors="ignore")
        .dropna()
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )

    correlation_selected = [
        str(column) for column in target_correlations.head(k_features).index
    ]

    correlation_ranking = [
        {
            "feature": str(column),
            "correlation": round(float(value), 4),
            "absoluteCorrelation": round(abs(float(value)), 4),
        }
        for column, value in target_correlations.items()
    ]

    rfe_selected = []
    rfe_ranking = []

    try:
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        estimator = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )

        rfe = RFE(
            estimator=estimator,
            n_features_to_select=k_features,
            step=1,
        )

        rfe.fit(X_scaled, y)

        rfe_selected = [
            str(feature)
            for feature, selected in zip(X.columns, rfe.support_)
            if selected
        ]

        rfe_ranking = sorted(
            [
                {
                    "feature": str(feature),
                    "rank": int(rank),
                    "selected": bool(selected),
                }
                for feature, rank, selected in zip(
                    X.columns, rfe.ranking_, rfe.support_
                )
            ],
            key=lambda item: item["rank"],
        )

    except Exception:
        rfe_selected = correlation_selected.copy()
        rfe_ranking = [
            {
                "feature": feature,
                "rank": index + 1,
                "selected": index < k_features,
            }
            for index, feature in enumerate(correlation_selected)
        ]

    feature_corr_matrix = X.corr(numeric_only=True).abs()
    mrmr_scores = []

    for feature in X.columns:
        relevance = abs(float(target_correlations.get(feature, 0.0)))
        other_features = [other for other in X.columns if other != feature]

        if other_features:
            redundancy = float(feature_corr_matrix.loc[feature, other_features].mean())
        else:
            redundancy = 0.0

        mrmr_score = relevance - redundancy

        mrmr_scores.append(
            {
                "feature": str(feature),
                "relevance": round(relevance, 4),
                "redundancy": round(redundancy, 4),
                "mrmrScore": round(float(mrmr_score), 4),
            }
        )

    mrmr_ranking = sorted(
        mrmr_scores,
        key=lambda item: item["mrmrScore"],
        reverse=True,
    )

    mrmr_selected = [item["feature"] for item in mrmr_ranking[:k_features]]

    redundant_pairs = []

    if X.shape[1] > 1:
        corr_matrix = X.corr(numeric_only=True).abs()

        for i, column_a in enumerate(corr_matrix.columns):
            for column_b in corr_matrix.columns[i + 1:]:
                corr_value = corr_matrix.loc[column_a, column_b]

                if corr_value >= 0.90:
                    redundant_pairs.append(
                        {
                            "featureA": str(column_a),
                            "featureB": str(column_b),
                            "correlation": round(float(corr_value), 4),
                            "suggestion": f"{column_b} değişkeni redundancy açısından incelenebilir.",
                        }
                    )

    recommended_removal = sorted(
        list(
            {
                item["featureB"]
                for item in redundant_pairs
                if item["featureB"] not in correlation_selected[:5]
            }
        )
    )

    method_sets = {
        "Correlation FS": set(correlation_selected),
        "RFE": set(rfe_selected),
        "mRMR-inspired": set(mrmr_selected),
    }

    all_selected_features = set().union(*method_sets.values())

    consensus_scores = []

    for feature in all_selected_features:
        vote_count = sum(
            1 for selected_set in method_sets.values() if feature in selected_set
        )

        consensus_scores.append(
            {
                "feature": feature,
                "votes": vote_count,
                "selectedBy": [
                    method
                    for method, selected_set in method_sets.items()
                    if feature in selected_set
                ],
            }
        )

    consensus_scores = sorted(
        consensus_scores,
        key=lambda item: item["votes"],
        reverse=True,
    )

    consensus_selected_features = [
        item["feature"] for item in consensus_scores if item["votes"] >= 2
    ]

    method_comparison = [
        {
            "method": "Correlation FS",
            "selectedFeatureCount": len(correlation_selected),
            "interpretability": "High",
            "redundancyControl": "Moderate",
            "stability": "High",
            "thesisCompatibility": "Very High",
            "selectedFeatures": correlation_selected,
        },
        {
            "method": "RFE",
            "selectedFeatureCount": len(rfe_selected),
            "interpretability": "Moderate",
            "redundancyControl": "Model-dependent",
            "stability": "Moderate",
            "thesisCompatibility": "Moderate",
            "selectedFeatures": rfe_selected,
        },
        {
            "method": "mRMR-inspired",
            "selectedFeatureCount": len(mrmr_selected),
            "interpretability": "High",
            "redundancyControl": "High",
            "stability": "Moderate",
            "thesisCompatibility": "High",
            "selectedFeatures": mrmr_selected,
        },
    ]

    report = {
        "targetColumn": "Class",
        "totalNumericFeatures": int(X.shape[1]),
        "kFeatures": int(k_features),
        "recommendedMethod": "Correlation FS",
        "recommendedReason": (
            "Correlation-based feature selection was selected as the final strategy "
            "because it is stable, interpretable, computationally transparent, and "
            "compatible with the proposed hybrid stacking framework."
        ),
        "correlationFS": {
            "method": "Correlation-based Feature Selection",
            "selectedFeatures": correlation_selected,
            "ranking": correlation_ranking,
        },
        "rfeFS": {
            "method": "Recursive Feature Elimination",
            "selectedFeatures": rfe_selected,
            "ranking": rfe_ranking,
        },
        "mrmrFS": {
            "method": "mRMR-inspired Feature Selection",
            "selectedFeatures": mrmr_selected,
            "ranking": mrmr_ranking,
        },
        "methodComparison": method_comparison,
        "consensusSelectedFeatures": consensus_selected_features,
        "consensusScores": consensus_scores,
        "redundantPairs": redundant_pairs[:15],
        "recommendedRemoval": recommended_removal,
        "recommendations": [
            "Correlation FS, RFE ve mRMR-inspired yöntemleri ayrı ayrı değerlendirilmiştir.",
            "Nihai yöntem olarak correlation-based feature selection önerilmiştir; çünkü tezde kullanılan hibrit stacking mimarisiyle daha yorumlanabilir ve daha kararlı bir yapı sunmaktadır.",
            "RFE modeli bağımlı bir seçim mekanizması sunduğu için destekleyici karşılaştırma yöntemi olarak değerlendirilmiştir.",
            "mRMR-inspired yaklaşım, hedef değişken ilişkisini korurken redundant değişkenleri azaltmaya yönelik tamamlayıcı bir analiz sağlamaktadır.",
            "Consensus feature listesi, birden fazla yöntem tarafından seçilen değişkenleri göstererek modelleme öncesi karar desteği sağlar.",
        ],
    }

    return {
        "status": "success",
        "message": "Feature Selector raporu başarıyla üretildi.",
        "report": report,
    }




def build_model_optimizer_report(df: pd.DataFrame, language: str = "tr"):
    if "Class" in df.columns:
        counts = df["Class"].value_counts(dropna=False).to_dict()
        class_distribution = {str(k): int(v) for k, v in counts.items()}
        majority = max(counts.values())
        minority = min(counts.values())
        imbalance_ratio = round(float(majority / minority), 4) if minority else None
    else:
        class_distribution = {}
        imbalance_ratio = None

    if language == "tr":
        imbalance_note = (
            f"Gözlenen sınıf dengesizliği oranı yaklaşık {imbalance_ratio}:1 düzeyindedir."
            if imbalance_ratio else "Class kolonu bulunamadı."
        )

        report = {
            "recommendedArchitecture": {
                "name": "Kalibre Edilmiş Hibrit Stacking Mimarisi",
                "featureSelection": "Korelasyon Tabanlı Özellik Seçimi",
                "preprocessing": [
                    "IQR tabanlı aykırı değer sınırlandırma",
                    "Standart ölçekleme",
                    "Modelleme öncesi eksik değer kontrolü",
                ],
                "baseLearners": [
                    "XGBoost Sınıflandırıcı",
                    "Random Forest Sınıflandırıcı",
                    "Bagging Sınıflandırıcı",
                ],
                "metaLearner": "Random Forest Meta-Öğrenici",
                "calibration": "Out-of-fold olasılık üretimi ile CalibratedClassifierCV kullanılmıştır.",
                "thresholdStrategy": "Optimize edilmiş hile riski karar eşiği",
            },
            "testSetPerformance": {
                "baseHybridFixedThreshold": {
                    "architecture": "Base Hybrid",
                    "threshold": 0.335,
                    "accuracy": 0.7333,
                    "precisionWeighted": 0.7523,
                    "recallWeighted": 0.7333,
                    "f1Weighted": 0.7388,
                    "balancedAccuracy": 0.7278,
                    "appGear": 0.7361,
                    "fraudPrecision": 0.5818,
                },
                "optimizedOOFStacking": {
                    "architecture": "OOF Stacking, Meta=RF, probs_only, local tuned",
                    "threshold": "Optimize",
                    "accuracy": 0.7630,
                    "precisionWeighted": 0.7811,
                    "recallWeighted": 0.7630,
                    "f1Weighted": 0.7678,
                    "balancedAccuracy": 0.7611,
                    "appGear": 0.7654,
                    "fraudPrecision": 0.6182,
                    "fraudRecall": 0.7556,
                },
                "note": "Bu değerler, tez kapsamında raporlanan kontrollü hibrit modelleme yapılandırmasının test seti performansını özetlemektedir.",
            },
            "baselineModelComparison": [
                {
                    "model": "XGBoost",
                    "role": "Base learner",
                    "strength": "Finansal oranlardaki doğrusal olmayan örüntüleri yakalayabilir.",
                    "limitation": "Kalibrasyon ve eşik kontrolü olmadan aşırı öğrenmeye yatkın olabilir.",
                },
                {
                    "model": "Random Forest",
                    "role": "Base learner ve meta learner",
                    "strength": "Kararlı ensemble davranışı ve değişken önemini sağlam biçimde değerlendirme avantajı sunar.",
                    "limitation": "Basit doğrusal tarama yöntemlerine göre daha az şeffaftır.",
                },
                {
                    "model": "Bagging",
                    "role": "Base learner",
                    "strength": "Bootstrap aggregation yoluyla varyansı azaltır.",
                    "limitation": "Karmaşık sinyallerde boosted modellere göre daha zayıf kalabilir.",
                },
            ],
            "thresholdOptimization": {
                "defaultThreshold": 0.50,
                "optimizedThreshold": 0.335,
                "rationale": "Hile tespiti, varsayılan 0.50 olasılık eşiğine dayanmak yerine fraud recall ve precision arasında kontrollü bir denge gerektirdiği için optimize edilmiş threshold tercih edilmiştir.",
                "searchRange": "0.30–0.41",
                "searchStep": 0.005,
            },
            "classImbalanceStrategy": {
                "classDistribution": class_distribution,
                "diagnosis": imbalance_note,
                "recommendation": "Koruyucu ve kontrollü bir sınıf dengesizliği stratejisi önerilir. Agresif sentetik dengeleme yöntemleri çapraz doğrulama skorlarını şişirebilir ve dış test seti kararlılığını azaltabilir.",
            },
            "calibrationReliability": {
                "strategy": "Out-of-fold kalibre edilmiş olasılık üretimi",
                "purpose": "Stacking ve threshold optimizasyonu öncesinde olasılık güvenilirliğini artırmak.",
                "recommendation": "Son karar destek katmanı hile olasılığı ve güven skoru raporladığı için kalibrasyon korunmalıdır.",
            },
            "hyperparameterStrategy": {
                "searchType": "Kontrollü yerel grid search",
                "metaLearnerCandidate": "Random Forest",
                "evaluatedParameters": [
                    "n_estimators",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "max_features",
                    "class_weight",
                ],
                "recommendation": "Test seti genellenebilirliğini korumak için agresif ve geniş arama yerine kontrollü yerel tuning tercih edilmelidir.",
            },
            "optimizerRecommendations": [
                "Tez mimarisiyle uyumlu nihai özellik taraması için korelasyon tabanlı özellik seçimi kullanılmalıdır.",
                "Meta-öğrenme öncesinde kalibre edilmiş out-of-fold olasılıklar korunmalıdır.",
                "Tekil base learner yerine hibrit stacking yapısı tercih edilmelidir.",
                "Varsayılan 0.50 threshold yerine optimize edilmiş karar eşiği kullanılmalıdır.",
                "Dış test seti kararlılığı doğrulanmadan agresif sentetik dengelemeden kaçınılmalıdır.",
                "Hile riski kararını etkileyen finansal oranları açıklamak için XAI aşamasına geçilmelidir.",
            ],
            "finalDecision": {
                "recommendedPipeline": [
                    "Korelasyon tabanlı özellik seçimi",
                    "IQR aykırı değer sınırlandırma",
                    "Standart ölçekleme",
                    "Kalibre edilmiş XGB + RF + Bagging",
                    "Random Forest meta-öğrenici",
                    "Optimize Threshold = 0.335",
                    "XAI destekli karar yorumu",
                ],
                "summary": "Kalibre edilmiş hibrit stacking mimarisi; kontrollü, yorumlanabilir ve tez mimarisiyle uyumlu bir hile tespit pipeline’ı sunduğu için nihai modelleme stratejisi olarak önerilmektedir.",
            },
        }
    else:
        report = {
            "recommendedArchitecture": {
                "name": "Calibrated Hybrid Stacking Framework",
                "featureSelection": "Correlation-based Feature Selection",
                "preprocessing": ["IQR-based Outlier Capping", "Standard Scaling", "Missing-value control before modeling"],
                "baseLearners": ["XGBoost Classifier", "Random Forest Classifier", "Bagging Classifier"],
                "metaLearner": "Random Forest Meta-Learner",
                "calibration": "CalibratedClassifierCV with out-of-fold probability generation",
                "thresholdStrategy": "Optimized fraud-risk decision threshold",
            },
            "testSetPerformance": {
                "baseHybridFixedThreshold": {
                    "architecture": "Base Hybrid",
                    "threshold": 0.335,
                    "accuracy": 0.7333,
                    "precisionWeighted": 0.7523,
                    "recallWeighted": 0.7333,
                    "f1Weighted": 0.7388,
                    "balancedAccuracy": 0.7278,
                    "appGear": 0.7361,
                    "fraudPrecision": 0.5818,
                },
                "optimizedOOFStacking": {
                    "architecture": "OOF Stacking, Meta=RF, probs_only, local tuned",
                    "threshold": "Optimized",
                    "accuracy": 0.7630,
                    "precisionWeighted": 0.7811,
                    "recallWeighted": 0.7630,
                    "f1Weighted": 0.7678,
                    "balancedAccuracy": 0.7611,
                    "appGear": 0.7654,
                    "fraudPrecision": 0.6182,
                    "fraudRecall": 0.7556,
                },
                "note": "Values summarize the thesis-reported test-set performance of the conservative hybrid modeling configuration.",
            },
            "baselineModelComparison": [
                {"model": "XGBoost", "role": "Base learner", "strength": "Captures nonlinear financial-ratio patterns", "limitation": "May overfit if used without calibration and threshold control"},
                {"model": "Random Forest", "role": "Base learner and meta learner", "strength": "Stable ensemble behavior and robust ranking of variables", "limitation": "Less transparent than simple linear screening methods"},
                {"model": "Bagging", "role": "Base learner", "strength": "Reduces variance through bootstrap aggregation", "limitation": "Can be weaker than boosted models on complex signals"},
            ],
            "thresholdOptimization": {
                "defaultThreshold": 0.50,
                "optimizedThreshold": 0.335,
                "rationale": "The optimized threshold is preferred because fraud detection requires a controlled balance between fraud recall and precision instead of relying on the default 0.50 probability cut-off.",
                "searchRange": "0.30–0.41",
                "searchStep": 0.005,
            },
            "classImbalanceStrategy": {
                "classDistribution": class_distribution,
                "diagnosis": f"Observed class imbalance ratio is approximately {imbalance_ratio}:1." if imbalance_ratio else "Class column was not found.",
                "recommendation": "A conservative imbalance strategy is recommended. Aggressive synthetic balancing may inflate cross-validation scores while reducing external test stability.",
            },
            "calibrationReliability": {
                "strategy": "Out-of-fold calibrated probability generation",
                "purpose": "To improve probability reliability before stacking and threshold optimization.",
                "recommendation": "Calibration should be preserved because the final decision-support layer reports fraud probability and confidence scores.",
            },
            "hyperparameterStrategy": {
                "searchType": "Local controlled grid search",
                "metaLearnerCandidate": "Random Forest",
                "evaluatedParameters": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "class_weight"],
                "recommendation": "Local tuning is preferred over aggressive broad search to preserve test-set generalization stability.",
            },
            "optimizerRecommendations": [
                "Use correlation-based feature selection as the final thesis-compatible feature screening strategy.",
                "Preserve calibrated out-of-fold probabilities before meta-learning.",
                "Prefer the hybrid stacking framework over a single base learner for stronger generalization behavior.",
                "Use threshold optimization instead of the default 0.50 decision threshold.",
                "Avoid aggressive synthetic balancing unless external test stability is explicitly verified.",
                "Proceed to XAI analysis to explain which financial ratios drive the fraud-risk decision.",
            ],
            "finalDecision": {
                "recommendedPipeline": ["Correlation-based Feature Selection", "IQR Outlier Capping", "Standard Scaling", "Calibrated XGB + RF + Bagging", "Random Forest Meta-Learner", "Optimized Threshold = 0.335", "XAI-supported decision interpretation"],
                "summary": "The calibrated hybrid stacking framework is recommended as the final modeling strategy because it provides a controlled, interpretable, and thesis-compatible fraud detection pipeline.",
            },
        }

    return {
        "status": "success",
        "message": "Model Optimizer raporu başarıyla üretildi.",
        "report": report,
    }


def build_xai_agent_report(df: pd.DataFrame, language: str = 'tr'):
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if "Class" in numeric_df.columns:
        feature_df = numeric_df.drop(columns=["Class"], errors="ignore")
        target_correlations = (
            numeric_df.corr(numeric_only=True)["Class"]
            .drop(labels=["Class"], errors="ignore")
            .dropna()
            .sort_values(key=lambda values: values.abs(), ascending=False)
        )
    else:
        feature_df = numeric_df
        target_correlations = pd.Series(dtype=float)

    top_influential_features = []

    for feature, corr in target_correlations.head(10).items():
        if language == "tr":
            direction = "Pozitif risk ilişkisi" if corr > 0 else "Negatif risk ilişkisi"
            interpretation = (
                f"{feature} değişkeni, hedef sınıf ile korelasyon taramasına göre "
                f"{direction.lower()} göstermektedir."
            )
        else:
            direction = "Positive risk association" if corr > 0 else "Negative risk association"
            interpretation = (
                f"{feature} shows a {direction.lower()} with the fraud class "
                "based on target-correlation screening."
            )

        top_influential_features.append(
            {
                "feature": str(feature),
                "importanceProxy": round(abs(float(corr)), 4),
                "correlation": round(float(corr), 4),
                "direction": direction,
                "interpretation": interpretation,
            }
        )

    if language == "tr":
        report = {
            "explainabilityMethod": {
                "name": "Korelasyon destekli XAI açıklama yaklaşımı",
                "purpose": "Model düzeyinde SHAP entegrasyonu öncesinde, en etkili finansal oranları analist tarafından anlaşılabilir biçimde açıklamak.",
                "scope": "Veri seti düzeyinde açıklama ve karar destek yorumu",
            },
            "topInfluentialFeatures": top_influential_features,
            "riskNarrative": {
                "summary": "XAI Agent, hile sınıfı ile en ilişkili finansal oranları belirleyerek bunları analist tarafından okunabilir risk göstergelerine dönüştürür.",
                "analystInterpretation": "Hedef sınıf ile daha güçlü mutlak ilişki gösteren değişkenler finansal tablo incelemesinde öncelikli olarak değerlendirilmelidir; ancak tek başına nedensel kanıt olarak yorumlanmamalıdır.",
                "decisionSupportRole": "Açıklama katmanı, model tahmininden sonra hangi oranların daha yakından incelenmesi gerektiğini göstererek insan denetimini destekler.",
            },
            "confidenceInterpretation": {
                "lowRisk": "Düşük hile olasılığı, gözlenen finansal oranların model tarafından öğrenilen hilesiz profile daha yakın olduğunu gösterir.",
                "highRisk": "Yüksek hile olasılığı, gözlenen finansal oranların hile sınıfı ile ilişkili örüntülere benzediğini gösterir.",
                "caution": "Olasılık değerleri kesin denetim sonucu olarak değil, karar destek sinyali olarak yorumlanmalıdır.",
            },
            "xaiRecommendations": [
                "En yüksek sıralanan oranlar, orijinal finansal tablo kalemleriyle birlikte incelenmelidir.",
                "Açıklama katmanı, bağımsız karar verici olarak değil, analist yargısını destekleyen bir mekanizma olarak kullanılmalıdır.",
                "Nihai hile riski değerlendirmesi yapılmadan önce model olasılığı, threshold kararı ve oran düzeyindeki açıklamalar birlikte ele alınmalıdır.",
                "Gelecek çalışmalarda şirket düzeyinde yerel açıklamalar için SHAP tabanlı açıklanabilirlik katmanı entegre edilebilir.",
            ],
            "finalExplanation": {
                "title": "Açıklanabilir Hile Riski Karar Desteği",
                "summary": "Yarı-ajan katmanı, model çıktılarını yorumlanabilir finansal oran sinyalleri ve analist tarafından okunabilir önerilerle ilişkilendirerek şeffaflığı artırır.",
            },
        }
    else:
        report = {
            "explainabilityMethod": {
                "name": "Correlation-guided XAI proxy explanation",
                "purpose": "To provide an interpretable analyst-facing explanation of the most influential financial ratios before model-level SHAP integration.",
                "scope": "Dataset-level explanation and decision-support interpretation",
            },
            "topInfluentialFeatures": top_influential_features,
            "riskNarrative": {
                "summary": "The XAI Agent highlights the financial ratios most associated with the fraud class and translates them into analyst-readable risk indicators.",
                "analystInterpretation": "Variables with stronger absolute association to the target class should be prioritized during financial statement review, but they should not be interpreted as standalone causal evidence.",
                "decisionSupportRole": "The explanation layer supports human review by showing which ratios require closer inspection after model prediction.",
            },
            "confidenceInterpretation": {
                "lowRisk": "Lower fraud probability indicates that the observed financial ratios are closer to the non-fraud profile learned by the model.",
                "highRisk": "Higher fraud probability indicates that the observed financial ratios resemble patterns associated with the fraud class.",
                "caution": "Probability values should be interpreted as decision-support signals, not as definitive audit conclusions.",
            },
            "xaiRecommendations": [
                "Review the highest-ranked ratios together with the original financial statement items.",
                "Use the explanation layer as a support mechanism for analyst judgment rather than an autonomous decision maker.",
                "Combine model probability, threshold decision, and ratio-level interpretation before producing a final fraud-risk conclusion.",
                "Future work may integrate SHAP-based local explanations for company-level prediction interpretation.",
            ],
            "finalExplanation": {
                "title": "Explainable Fraud-Risk Decision Support",
                "summary": "The semi-agentic layer improves transparency by connecting model outputs with interpretable financial-ratio signals and analyst-readable recommendations.",
            },
        }

    return {
        "status": "success",
        "message": "XAI Agent raporu başarıyla üretildi.",
        "report": report,
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


@app.post("/agentic/upload-preview")
async def agentic_upload_preview(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {
            "status": "error",
            "message": "Lütfen CSV formatında bir dosya yükleyin.",
        }

    try:
        df = pd.read_csv(file.file)
        return build_upload_preview(df)

    except Exception as error:
        return {
            "status": "error",
            "message": f"CSV dosyası okunamadı: {str(error)}",
        }


@app.post("/agentic/data-analyst")
async def run_data_analyst(file: UploadFile = File(...), instruction: str = Form("")):
    if not file.filename.endswith(".csv"):
        return {
            "status": "error",
            "message": "Lütfen CSV formatında bir dosya yükleyin.",
        }

    try:
        df = pd.read_csv(file.file)
        return build_data_analyst_report(df, instruction)

    except Exception as error:
        return {
            "status": "error",
            "message": f"Data Analyst raporu üretilemedi: {str(error)}",
        }


@app.post("/agentic/feature-selector")
async def run_feature_selector(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {
            "status": "error",
            "message": "Lütfen CSV formatında bir dosya yükleyin.",
        }

    try:
        df = pd.read_csv(file.file)
        return build_feature_selector_report(df)

    except Exception as error:
        return {
            "status": "error",
            "message": f"Feature Selector raporu üretilemedi: {str(error)}",
        }
    

@app.post("/agentic/model-optimizer")
async def run_model_optimizer(
    file: UploadFile = File(...),
    language: str = Form("tr"),
):
    if not file.filename.endswith(".csv"):
        return {
            "status": "error",
            "message": "Lütfen CSV formatında bir dosya yükleyin.",
        }

    try:
        df = pd.read_csv(file.file)
        return build_model_optimizer_report(df, language=language)

    except Exception as error:
        return {
            "status": "error",
            "message": f"Model Optimizer raporu üretilemedi: {str(error)}",
        }



@app.post("/agentic/xai-agent")
async def run_xai_agent(
    file: UploadFile = File(...),
    language: str = Form("tr"),
):
    if not file.filename.endswith(".csv"):
        return {
            "status": "error",
            "message": "Lütfen CSV formatında bir dosya yükleyin.",
        }

    try:
        df = pd.read_csv(file.file)
        return build_xai_agent_report(df, language=language)

    except Exception as error:
        return {
            "status": "error",
            "message": f"XAI Agent raporu üretilemedi: {str(error)}",
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