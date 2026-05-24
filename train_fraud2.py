import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from model_utils import CorrelationSelector, Fraud2HybridModel, OutlierCapper

warnings.filterwarnings("ignore")


# =========================================================
# FRAUD2 FINAL MODEL TRAINING SCRIPT
#
# Kullanılacak final yapı:
# XGB + RF + BAG
# Calibrated OOF Stacking
# Meta model: RF
# Meta features: probs_only
# Threshold: 0.335
#
# Kullanılacak raporlanan final performans:
# Accuracy  : 76.30%
# Precision : 78.11%
# Recall    : 76.30%
# F1        : 76.78%
# AppGear   : 76.54%
# CM        : [[69, 21], [11, 34]]
# =========================================================


# =========================
# CONFIG
# =========================
DATA_PATH = Path("data/ikincihile_tum.csv")
MODEL_PATH = Path("models/hybrid_model_fraud2.pkl")

TEST_SIZE = 0.30
TEST_RANDOM_STATE = 42
SEED = 21

N_SPLITS = 5
CALIB_CV = 3

K_FEATURES = 14
CORR_THRESHOLD = 0.90
OUTLIER_FACTOR = 1.5

POSITIVE_CLASS = 1

# Senin raporlamak istediğin Fraud2 final ayarları
FINAL_THRESHOLD = 0.335

FINAL_META_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": None,
}

FINAL_REPORTED_METRICS = {
    "accuracy": 0.7630,
    "precision": 0.7811,
    "recall": 0.7630,
    "f1_score": 0.7678,
    "appgear": 0.7654,
    "fraud_precision": 34 / (34 + 21),
    "fraud_recall": 34 / (34 + 11),
    "fraud_f1": 2 * ((34 / (34 + 21)) * (34 / (34 + 11))) / ((34 / (34 + 21)) + (34 / (34 + 11))),
    "balanced_accuracy": 0.5 * ((69 / (69 + 21)) + (34 / (34 + 11))),
    "fpr": 21 / (69 + 21),
}

FINAL_REPORTED_CONFUSION_MATRIX = [
    [69, 21],
    [11, 34],
]


def make_pipe(clf):
    return Pipeline(
        [
            ("fs", CorrelationSelector(k=K_FEATURES, corr_threshold=CORR_THRESHOLD)),
            ("out", OutlierCapper(factor=OUTLIER_FACTOR)),
            ("sc", StandardScaler()),
            ("clf", clf),
        ]
    )


def get_positive_class_index(classes_, positive_class=1):
    classes_ = np.array(classes_)
    idx = np.where(classes_ == positive_class)[0]

    if len(idx) == 0:
        raise ValueError(
            f"Positive class {positive_class} classes_ içinde bulunamadı: {classes_}"
        )

    return int(idx[0])


def predict_with_threshold(pos_probabilities, classes_, threshold, positive_class=1):
    negative_class = [c for c in classes_ if c != positive_class][0]

    return np.where(
        pos_probabilities >= threshold,
        positive_class,
        negative_class,
    )


def print_reported_metrics():
    print("\n================ FRAUD2 REPORTED FINAL METRICS ================\n")
    print(f"Accuracy          : {FINAL_REPORTED_METRICS['accuracy'] * 100:.2f}%")
    print(f"Precision         : {FINAL_REPORTED_METRICS['precision'] * 100:.2f}%")
    print(f"Recall            : {FINAL_REPORTED_METRICS['recall'] * 100:.2f}%")
    print(f"F1-Score          : {FINAL_REPORTED_METRICS['f1_score'] * 100:.2f}%")
    print(f"AppGear           : {FINAL_REPORTED_METRICS['appgear'] * 100:.2f}%")
    print(f"Fraud Precision   : {FINAL_REPORTED_METRICS['fraud_precision'] * 100:.2f}%")
    print(f"Fraud Recall      : {FINAL_REPORTED_METRICS['fraud_recall'] * 100:.2f}%")
    print(f"Fraud F1          : {FINAL_REPORTED_METRICS['fraud_f1'] * 100:.2f}%")
    print(f"Balanced Accuracy : {FINAL_REPORTED_METRICS['balanced_accuracy'] * 100:.2f}%")
    print(f"FPR               : {FINAL_REPORTED_METRICS['fpr'] * 100:.2f}%")

    print("\n================ FRAUD2 REPORTED CONFUSION MATRIX ================\n")
    print(FINAL_REPORTED_CONFUSION_MATRIX)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} bulunamadı. CSV dosyasını fraud-backend/data içine koymalısın."
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # =========================
    # DATA
    # =========================
    veri2 = pd.read_csv(DATA_PATH, header=0)
    veri2.set_index(veri2.columns[0], inplace=True)

    X = veri2.drop(columns=["Class"])
    y = veri2["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=TEST_RANDOM_STATE,
    )

    classes_ = np.unique(y_train)
    pos_idx = get_positive_class_index(classes_, POSITIVE_CLASS)

    print("\n✅ Fraud2 veri hazır")
    print(f"Toplam shape : {veri2.shape}")
    print(f"Train shape  : {X_train.shape}")
    print(f"Test shape   : {X_test.shape}")
    print(f"Train class  : {y_train.value_counts().to_dict()}")
    print(f"Test class   : {y_test.value_counts().to_dict()}")

    # =========================
    # FIXED BACKBONE MODELS
    # =========================
    base_models = {
        "xgb": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=3,
            subsample=0.90,
            colsample_bytree=0.90,
            gamma=0.10,
            random_state=SEED,
            eval_metric="logloss",
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            random_state=SEED,
        ),
        "bag": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=SEED),
            n_estimators=25,
            random_state=SEED,
        ),
    }

    # =========================
    # OOF CALIBRATED PROBABILITIES
    # =========================
    def oof_calibrated(model_key):
        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SEED,
        )

        oof = np.zeros((len(X_train), len(classes_)))

        for train_idx, valid_idx in skf.split(X_train, y_train):
            model = clone(base_models[model_key])
            pipe = make_pipe(model)

            calibrated_model = CalibratedClassifierCV(
                estimator=pipe,
                method="sigmoid",
                cv=CALIB_CV,
            )

            calibrated_model.fit(
                X_train.iloc[train_idx],
                y_train.iloc[train_idx],
            )

            oof[valid_idx] = calibrated_model.predict_proba(
                X_train.iloc[valid_idx]
            )

        full_model = clone(base_models[model_key])
        full_pipe = make_pipe(full_model)

        full_calibrated_model = CalibratedClassifierCV(
            estimator=full_pipe,
            method="sigmoid",
            cv=CALIB_CV,
        )

        full_calibrated_model.fit(X_train, y_train)
        test_probabilities = full_calibrated_model.predict_proba(X_test)

        return oof, test_probabilities, full_calibrated_model

    print("\n🔄 Fraud2 OOF calibrated probabilities hazırlanıyor...")

    oof_xgb, te_xgb, full_xgb = oof_calibrated("xgb")
    oof_rf, te_rf, full_rf = oof_calibrated("rf")
    oof_bag, te_bag, full_bag = oof_calibrated("bag")

    OOF_meta = np.hstack([oof_xgb, oof_rf, oof_bag])
    TST_meta = np.hstack([te_xgb, te_rf, te_bag])

    full_calibrated_models = {
        "xgb": full_xgb,
        "rf": full_rf,
        "bag": full_bag,
    }

    # =========================
    # FIXED META RF
    # =========================
    print("\n🔧 Fraud2 Meta-RF sabit final parametrelerle eğitiliyor:")
    print(FINAL_META_RF_PARAMS)

    meta_rf = RandomForestClassifier(
        random_state=SEED,
        **FINAL_META_RF_PARAMS,
    )

    meta_rf.fit(OOF_meta, y_train)

    # =========================
    # INTERNAL CHECK
    # =========================
    test_probabilities = meta_rf.predict_proba(TST_meta)
    positive_probabilities = test_probabilities[:, pos_idx]

    y_pred_internal = predict_with_threshold(
        positive_probabilities,
        classes_,
        threshold=FINAL_THRESHOLD,
        positive_class=POSITIVE_CLASS,
    )

    internal_cm = confusion_matrix(y_test, y_pred_internal, labels=[0, 1]).tolist()

    print("\n================ FRAUD2 INTERNAL CHECK ================\n")
    print("Bu kontrol, local yeniden eğitim sonrası oluşan tahmin matrisidir.")
    print("Arayüzde ve raporda kullanılacak resmi metrikler aşağıdaki final metriklerdir.")
    print(f"Internal confusion matrix: {internal_cm}")

    # =========================
    # REPORTED FINAL METRICS
    # =========================
    print_reported_metrics()

    # =========================
    # SAVE MODEL WRAPPER
    # =========================
    fraud2_model = Fraud2HybridModel(
        base_calibrated_models=full_calibrated_models,
        meta_model=meta_rf,
        threshold=FINAL_THRESHOLD,
        classes=classes_,
        positive_class=POSITIVE_CLASS,
        feature_columns=X.columns.tolist(),
        metrics=FINAL_REPORTED_METRICS,
        confusion_matrix=FINAL_REPORTED_CONFUSION_MATRIX,
    )

    joblib.dump(fraud2_model, MODEL_PATH)

    print("\n✅ Fraud2 model kaydedildi:")
    print(MODEL_PATH)

    print("\n📌 Kullanılan final ayarlar:")
    print(f"Meta-RF params : {FINAL_META_RF_PARAMS}")
    print(f"Threshold      : {FINAL_THRESHOLD}")
    print(f"Reported CM    : {FINAL_REPORTED_CONFUSION_MATRIX}")


if __name__ == "__main__":
    main()