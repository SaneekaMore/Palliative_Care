"""Machine learning utilities for palliative care risk prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


RISK_LABELS = {0: "Stable", 1: "Moderate", 2: "Critical"}


@dataclass
class ModelArtifacts:
    """Stores trained model and metadata."""

    model: RandomForestClassifier
    feature_names: list[str]


def _label_risk(row: pd.Series) -> int:
    """Rule-based labels used to generate synthetic training data.

    - Critical: SpO2 < 90 OR pain > 8 OR temp >= 39.3 OR heart rate > 125
    - Moderate: mild deviations
    - Stable: everything else
    """
    if row["spo2"] < 90 or row["pain"] > 8 or row["temp"] >= 39.3 or row["heart_rate"] > 125:
        return 2
    if (
        row["spo2"] < 94
        or row["pain"] >= 6
        or row["temp"] >= 37.8
        or row["heart_rate"] > 105
        or row["heart_rate"] < 62
    ):
        return 1
    return 0


def generate_synthetic_data(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic patient data for model training."""
    rng = np.random.default_rng(random_state)

    data = pd.DataFrame(
        {
            "heart_rate": rng.integers(60, 141, n_samples),
            "spo2": rng.integers(85, 101, n_samples),
            "temp": np.round(rng.uniform(36.0, 40.0, n_samples), 1),
            "pain": rng.integers(1, 11, n_samples),
        }
    )

    data["risk"] = data.apply(_label_risk, axis=1)
    return data


def train_model(random_state: int = 42) -> Tuple[ModelArtifacts, float]:
    """Train RandomForest model and return artifacts + validation accuracy."""
    df = generate_synthetic_data(random_state=random_state)
    feature_names = ["heart_rate", "spo2", "temp", "pain"]

    X = df[feature_names]
    y = df["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=10,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return ModelArtifacts(model=model, feature_names=feature_names), accuracy


def predict_risk(
    artifacts: ModelArtifacts,
    heart_rate: int,
    spo2: int,
    temp: float,
    pain: int,
) -> tuple[str, float, dict[str, float]]:
    """Predict risk label and probabilities."""
    input_df = pd.DataFrame(
        [[heart_rate, spo2, temp, pain]],
        columns=artifacts.feature_names,
    )

    probas = artifacts.model.predict_proba(input_df)[0]
    pred_idx = int(np.argmax(probas))
    probability_score = float(np.max(probas))

    prob_dict = {RISK_LABELS[i]: float(p) for i, p in enumerate(probas)}
    return RISK_LABELS[pred_idx], probability_score, prob_dict


def forecast_24h_trend(heart_rate: int, spo2: int, temp: float, pain: int) -> pd.DataFrame:
    """Generate a simple 24-hour trend forecast from current vitals."""
    hours = np.arange(0, 25)

    hr_trend = heart_rate + np.sin(hours / 4) * 4 + np.linspace(0, 2, len(hours))
    spo2_trend = spo2 + np.cos(hours / 6) * 1.2 - np.linspace(0, 1, len(hours)) * (1 if spo2 < 94 else 0.2)
    temp_trend = temp + np.sin(hours / 7) * 0.2 + (0.4 if temp >= 38 else 0.05) * (hours / 24)
    pain_trend = pain + np.sin(hours / 5) * 0.6 + (0.7 if pain >= 7 else -0.2) * (hours / 24)

    trend = pd.DataFrame(
        {
            "hour": hours,
            "Heart Rate": np.round(np.clip(hr_trend, 55, 150), 1),
            "SpO2": np.round(np.clip(spo2_trend, 82, 100), 1),
            "Temperature": np.round(np.clip(temp_trend, 35.5, 40.5), 1),
            "Pain": np.round(np.clip(pain_trend, 1, 10), 1),
        }
    )
    return trend
