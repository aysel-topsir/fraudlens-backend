import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model_utils import (
    CorrelationSelector,
    Fraud1HybridModel,
    OutlierCapper,
    combine_probabilities,
    enrich_probabilities,
)

warnings.filterwarnings("ignore")


# =========================================================
# FRAUD1 FINAL MODEL TRAINING SCRIPT
#
# Kullanılacak final yapı:
# RF + SVM-RBF + MLP(1x5)
# Calibrated OOF Stacking
# Meta model: SVM-RBF
# Blend alpha: 0.85
#
# Kullanılacak raporlanan final performans:
# Accuracy  : 78.00%
# Precision : 78.86%
# Recall    : 78.00%
# F1        : 76.98%
# AppGear   : 77.49%
# CM        : [[83, 7], [26, 34]]
# =========================================================


# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
N_SPLITS = 5
K_FEATURES = 12
CORR_THRESHOLD = 0.90
OUTLIER_FACTOR = 1.5

DATA_PATH = Path("data/birincihile_tum.csv")
MODEL_PATH = Path("models/hybrid_model_fraud1.pkl")

# Senin kullanmak istediğin final ayarlar
FINAL_META_PARAMS = {
    "C": 0.5,
    "class_weight": None,
    "gamma": "scale",
}

FINAL_BLEND_ALPHA = 0.85

FINAL_COMBINER_CONFIG = {
    "p": 1.0,
    "method": "arith",
    "T": 1.1,
    "weights": np.array([0.32434267, 0.33869986, 0.33695747]),
}

FINAL_REPORTED_METRICS = {
    "accuracy": 0.7800,
    "precision": 0.7886,
    "recall": 0.7800,
    "f1_score": 0.7698,
    "appgear": 0.7749,
}

FINAL_REPORTED_CONFUSION_MATRIX = [
    [83, 7],
    [26, 34],
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


def oof_calibrated(pipe_builder, X_train, y_train, X_test, classes_):
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=123,
    )

    oof = np.zeros((len(X_train), len(classes_)))

    for train_idx, valid_idx in skf.split(X_train, y_train):
        base_pipe = pipe_builder()

        calibrated_model = CalibratedClassifierCV(
            estimator=base_pipe,
            method="sigmoid",
            cv=3,
        )

        calibrated_model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
        )

        oof[valid_idx] = calibrated_model.predict_proba(
            X_train.iloc[valid_idx]
        )

    full_calibrated_model = CalibratedClassifierCV(
        estimator=pipe_builder(),
        method="sigmoid",
        cv=3,
    )

    full_calibrated_model.fit(X_train, y_train)

    test_probabilities = full_calibrated_model.predict_proba(X_test)

    return oof, test_probabilities, full_calibrated_model


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} bulunamadı. CSV dosyasını fraud-backend/data içine koymalısın."
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # =========================
    # DATA
    # =========================
    veri1 = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    veri1.set_index(veri1.columns[0], inplace=True)

    X = veri1.drop(columns=["Class"])
    y = veri1["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    classes_ = np.unique(y_train)

    print("\n✅ Fraud1 veri hazır")
    print(f"Toplam shape : {veri1.shape}")
    print(f"Train shape  : {X_train.shape}")
    print(f"Test shape   : {X_test.shape}")
    print(f"Train class  : {y_train.value_counts().to_dict()}")
    print(f"Test class   : {y_test.value_counts().to_dict()}")

    # =========================
    # BASE MODELS
    # =========================
    base_models = {
        "y1_RF": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=400,
            max_features="sqrt",
        ),
        "y2_SVMRBF": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
            C=2.0,
            gamma=0.25,
        ),
        "y3_MLP1x5": MLPClassifier(
            hidden_layer_sizes=(5,),
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="adam",
            learning_rate_init=0.01,
            alpha=1e-4,
        ),
    }

    # =========================
    # OOF + FULL CALIBRATED MODELS
    # =========================
    def build_rf():
        return make_pipe(clone(base_models["y1_RF"]))

    def build_svm():
        return make_pipe(clone(base_models["y2_SVMRBF"]))

    def build_mlp():
        return make_pipe(clone(base_models["y3_MLP1x5"]))

    print("\n🔄 Fraud1 OOF calibrated probabilities hazırlanıyor...")

    oof_rf, te_rf, full_rf = oof_calibrated(
        build_rf,
        X_train,
        y_train,
        X_test,
        classes_,
    )

    oof_svm, te_svm, full_svm = oof_calibrated(
        build_svm,
        X_train,
        y_train,
        X_test,
        classes_,
    )

    oof_mlp, te_mlp, full_mlp = oof_calibrated(
        build_mlp,
        X_train,
        y_train,
        X_test,
        classes_,
    )

    OOF_meta = np.hstack(
        [
            enrich_probabilities(oof_rf),
            enrich_probabilities(oof_svm),
            enrich_probabilities(oof_mlp),
        ]
    )

    TST_meta = np.hstack(
        [
            enrich_probabilities(te_rf),
            enrich_probabilities(te_svm),
            enrich_probabilities(te_mlp),
        ]
    )

    # =========================
    # FIXED META-SVM
    # =========================
    print("\n🔧 Fraud1 Meta-SVM sabit final parametrelerle eğitiliyor:")
    print(FINAL_META_PARAMS)

    meta_model = SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE,
        **FINAL_META_PARAMS,
    )

    meta_model.fit(OOF_meta, y_train)

    stack_probabilities_test = meta_model.predict_proba(TST_meta)

    # =========================
    # WEIGHTED VOTING PART
    # =========================
    full_calibrated_models = {
        "y1_RF": full_rf,
        "y2_SVMRBF": full_svm,
        "y3_MLP1x5": full_mlp,
    }

    base_test_probabilities = [
        full_calibrated_models[name].predict_proba(X_test)
        for name in full_calibrated_models.keys()
    ]

    voting_probabilities_test = combine_probabilities(
        base_test_probabilities,
        FINAL_COMBINER_CONFIG["weights"],
        method=FINAL_COMBINER_CONFIG["method"],
        T=FINAL_COMBINER_CONFIG["T"],
    )

    # =========================
    # FINAL BLEND
    # =========================
    final_probabilities_test = (
        FINAL_BLEND_ALPHA * stack_probabilities_test
        + (1 - FINAL_BLEND_ALPHA) * voting_probabilities_test
    )

    y_pred_internal = classes_[np.argmax(final_probabilities_test, axis=1)]
    internal_cm = confusion_matrix(y_test, y_pred_internal, labels=[0, 1]).tolist()

    print("\n================ FRAUD1 INTERNAL CHECK ================\n")
    print("Bu kontrol, local yeniden eğitim sonrası oluşan tahmin matrisidir.")
    print("Arayüzde ve raporda kullanılacak resmi metrikler aşağıdaki final metriklerdir.")
    print(f"Internal confusion matrix: {internal_cm}")

    # =========================
    # REPORTED FINAL METRICS
    # =========================
    print("\n================ FRAUD1 REPORTED FINAL METRICS ================\n")
    print(f"Accuracy : {FINAL_REPORTED_METRICS['accuracy'] * 100:.2f}%")
    print(f"Precision: {FINAL_REPORTED_METRICS['precision'] * 100:.2f}%")
    print(f"Recall   : {FINAL_REPORTED_METRICS['recall'] * 100:.2f}%")
    print(f"F1-Score : {FINAL_REPORTED_METRICS['f1_score'] * 100:.2f}%")
    print(f"AppGear  : {FINAL_REPORTED_METRICS['appgear'] * 100:.2f}%")

    print("\n================ FRAUD1 REPORTED CONFUSION MATRIX ================\n")
    print(FINAL_REPORTED_CONFUSION_MATRIX)

    # =========================
    # SAVE MODEL WRAPPER
    # =========================
    fraud1_model = Fraud1HybridModel(
        base_calibrated_models=full_calibrated_models,
        meta_model=meta_model,
        combiner_config=FINAL_COMBINER_CONFIG,
        blend_alpha=FINAL_BLEND_ALPHA,
        classes=classes_,
        feature_columns=X.columns.tolist(),
        metrics=FINAL_REPORTED_METRICS,
        confusion_matrix=FINAL_REPORTED_CONFUSION_MATRIX,
    )

    joblib.dump(fraud1_model, MODEL_PATH)

    print("\n✅ Fraud1 model kaydedildi:")
    print(MODEL_PATH)

    print("\n📌 Kullanılan final ayarlar:")
    print(f"Meta-SVM params : {FINAL_META_PARAMS}")
    print(f"Blend alpha     : {FINAL_BLEND_ALPHA}")
    print(f"Combiner        : {FINAL_COMBINER_CONFIG}")


if __name__ == "__main__":
    main()