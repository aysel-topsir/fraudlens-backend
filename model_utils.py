import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_ = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        for col in X_.select_dtypes(include=[np.number]).columns:
            q1 = X_[col].quantile(0.25)
            q3 = X_[col].quantile(0.75)
            iqr = q3 - q1

            lo = q1 - self.factor * iqr
            hi = q3 + self.factor * iqr

            if not np.isfinite(lo):
                lo = X_[col].min()
            if not np.isfinite(hi):
                hi = X_[col].max()
            if lo == hi:
                lo, hi = X_[col].min(), X_[col].max()

            self.bounds_[col] = (lo, hi)

        return self

    def transform(self, X):
        X_ = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()

        for col, (lo, hi) in self.bounds_.items():
            if col in X_.columns:
                X_[col] = X_[col].clip(lower=lo, upper=hi)

        return X_ if isinstance(X, pd.DataFrame) else X_.values


class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, corr_threshold=0.90):
        self.k = k
        self.corr_threshold = corr_threshold
        self.selected_columns_ = None
        self.selected_idx_ = None
        self.feature_scores_ = None

    def _target_corr_scores(self, X: pd.DataFrame, y):
        if not np.issubdtype(pd.Series(y).dtype, np.number):
            y_enc = LabelEncoder().fit_transform(y)
        else:
            y_enc = np.asarray(y)

        y_enc = pd.Series(y_enc, index=X.index)
        n_classes = len(np.unique(y_enc))

        if n_classes > 2:
            Y = pd.DataFrame(LabelBinarizer().fit_transform(y_enc), index=X.index)

            if Y.shape[1] == 1:
                Y = pd.DataFrame({0: Y.iloc[:, 0]}, index=X.index)

            scores = {}

            for col in X.columns:
                x = X[col]
                rmax = 0.0

                for c in Y.columns:
                    r = pd.concat([x, Y[c]], axis=1).corr().iloc[0, 1]

                    if pd.isna(r):
                        r = 0.0

                    rmax = max(rmax, abs(r))

                scores[col] = rmax

            return pd.Series(scores)

        s = X.apply(
            lambda s_: abs(pd.concat([s_, y_enc], axis=1).corr().iloc[0, 1]),
            axis=0,
        )

        return s.fillna(0.0)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        scores = self._target_corr_scores(X, y).sort_values(ascending=False)
        self.feature_scores_ = scores

        prelim = (
            list(scores.index[: min(self.k, X.shape[1])])
            if self.k is not None
            else list(scores.index)
        )

        selected = []
        corr_abs = X[prelim].corr().abs().fillna(0.0)

        for feat in prelim:
            if all(
                corr_abs.loc[feat, selected_feat] <= self.corr_threshold
                for selected_feat in selected
            ):
                selected.append(feat)

        self.selected_columns_ = selected
        self.selected_idx_ = [X.columns.get_loc(c) for c in selected]

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X.iloc[:, self.selected_idx_]


def appgear(acc, f1):
    return 0.5 * (acc + f1)


def softmax_T(p, T=1.0, axis=1):
    z = np.log(np.clip(p, 1e-12, 1.0)) / T
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def combine_probabilities(P_list, weights, method="arith", T=1.0):
    Ps = [softmax_T(P, T) for P in P_list]
    W = np.array(weights).reshape(-1, 1, 1)

    if method == "arith":
        P = np.sum(W * np.stack(Ps, 0), axis=0)

    elif method == "geom":
        P = np.exp(
            np.sum(
                W * np.log(np.clip(np.stack(Ps, 0), 1e-12, 1.0)),
                axis=0,
            )
        )

    elif method == "logit":
        eps = 1e-12
        L = [
            np.log(np.clip(P, eps, 1 - eps))
            - np.log(np.clip(1 - P, eps, 1 - eps))
            for P in Ps
        ]
        Lw = np.sum(W * np.stack(L, 0), axis=0)
        P = 1 / (1 + np.exp(-Lw))

    else:
        raise ValueError("unknown combiner")

    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)

    return P


def enrich_probabilities(P):
    eps = 1e-12
    maxp = P.max(1, keepdims=True)
    ent = -np.sum(P * np.log(np.clip(P, eps, 1.0)), axis=1, keepdims=True)
    sortp = -np.sort(-P, axis=1)
    margin = (sortp[:, 0] - sortp[:, 1]).reshape(-1, 1)

    return np.hstack([P, maxp, ent, margin])


class Fraud1HybridModel:
    def __init__(
        self,
        base_calibrated_models,
        meta_model,
        combiner_config,
        blend_alpha,
        classes,
        feature_columns,
        metrics,
        confusion_matrix,
    ):
        self.base_calibrated_models = base_calibrated_models
        self.meta_model = meta_model
        self.combiner_config = combiner_config
        self.blend_alpha = blend_alpha
        self.classes = np.array(classes)
        self.feature_columns = list(feature_columns)
        self.metrics = metrics
        self.confusion_matrix = confusion_matrix

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)

        X = X[self.feature_columns]

        model_names = list(self.base_calibrated_models.keys())

        base_probabilities = [
            self.base_calibrated_models[name].predict_proba(X)
            for name in model_names
        ]

        vote_probabilities = combine_probabilities(
            base_probabilities,
            self.combiner_config["weights"],
            method=self.combiner_config["method"],
            T=self.combiner_config["T"],
        )

        meta_features = np.hstack(
            [enrich_probabilities(P) for P in base_probabilities]
        )

        stack_probabilities = self.meta_model.predict_proba(meta_features)

        final_probabilities = (
            self.blend_alpha * stack_probabilities
            + (1 - self.blend_alpha) * vote_probabilities
        )

        return final_probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]


class Fraud2HybridModel:
    def __init__(
        self,
        base_calibrated_models,
        meta_model,
        threshold,
        classes,
        positive_class,
        feature_columns,
        metrics,
        confusion_matrix,
    ):
        self.base_calibrated_models = base_calibrated_models
        self.meta_model = meta_model
        self.threshold = float(threshold)
        self.classes = np.array(classes)
        self.positive_class = positive_class
        self.feature_columns = list(feature_columns)
        self.metrics = metrics
        self.confusion_matrix = confusion_matrix

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)

        X = X[self.feature_columns]

        model_names = list(self.base_calibrated_models.keys())

        base_probabilities = [
            self.base_calibrated_models[name].predict_proba(X)
            for name in model_names
        ]

        meta_features = np.hstack(base_probabilities)
        final_probabilities = self.meta_model.predict_proba(meta_features)

        return final_probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)

        positive_index = np.where(self.classes == self.positive_class)[0][0]
        positive_probabilities = probabilities[:, positive_index]

        negative_class = [c for c in self.classes if c != self.positive_class][0]

        return np.where(
            positive_probabilities >= self.threshold,
            self.positive_class,
            negative_class,
        )