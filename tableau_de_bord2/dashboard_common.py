from pathlib import Path

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_PATH = Path(__file__).parent.parent / "digitalTwin.csv"
BASE_FEATURES = ["HealthIndex", "EngineTemp", "Vibration", "Pressure"]
FEATURES = [
    "HealthIndex",
    "EngineTemp",
    "Vibration",
    "Pressure",
    "TempVibration",
    "PressureVibrationRatio",
    "HealthRisk",
]


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path).drop_duplicates().dropna().copy()
    dataframe[BASE_FEATURES] = dataframe[BASE_FEATURES].apply(pd.to_numeric, errors="coerce")
    dataframe = dataframe.dropna(subset=BASE_FEATURES)
    dataframe = dataframe[(dataframe[BASE_FEATURES] >= 0).all(axis=1)].copy()
    dataframe["TempVibration"] = dataframe["EngineTemp"] * dataframe["Vibration"]
    dataframe["PressureVibrationRatio"] = dataframe["Pressure"] / (dataframe["Vibration"] + 0.01)
    dataframe["HealthRisk"] = 100 - dataframe["HealthIndex"]
    return dataframe


@st.cache_resource
def train_model(model_name: str, csv_path: str, modified_at: float):
    dataframe = load_data(csv_path)
    X = dataframe[FEATURES]
    y = dataframe["State"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    if model_name == "CatBoost":
        model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
        )
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = Pipeline(
            [
                ("normalisation", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    model.fit(X, y)
    return model, accuracy, f1, len(dataframe), dataframe["Equipment"].nunique()


def run_dashboard(model_name: str) -> None:
    st.set_page_config(page_title=f"Jumeau numérique — {model_name}", page_icon="⚙️", layout="wide")
    st.title(f"Jumeau numérique minier — {model_name}")
    st.caption("Simulation d'un équipement et prédiction de son état : Healthy, Warning ou Critical.")

    if not DATA_PATH.exists():
        st.error(f"Dataset introuvable : {DATA_PATH.name}")
        st.stop()

    dataframe = load_data(str(DATA_PATH))
    with st.spinner(f"Entraînement de {model_name}..."):
        model, accuracy, f1, number_rows, number_equipment = train_model(
            model_name, str(DATA_PATH), DATA_PATH.stat().st_mtime
        )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Relevés", f"{number_rows:,}")
    metric_2.metric("Équipements", f"{number_equipment:,}")
    metric_3.metric("Accuracy test", f"{accuracy:.1%}")
    metric_4.metric("F1 pondéré", f"{f1:.3f}")

    st.divider()
    st.subheader("Simuler un relevé")
    col_1, col_2, col_3, col_4 = st.columns(4)
    health = col_1.slider("HealthIndex", 0.0, 100.0, 50.0, 0.01)
    temperature = col_2.slider("EngineTemp", 50.0, 120.0, 85.0, 0.01)
    vibration = col_3.slider("Vibration", 0.0, 10.0, 5.0, 0.01)
    pressure = col_4.slider("Pressure", 100.0, 350.0, 225.0, 0.01)

    observation = pd.DataFrame(
        [[health, temperature, vibration, pressure, temperature * vibration,
          pressure / (vibration + 0.01), 100 - health]],
        columns=FEATURES,
    )
    probabilities = model.predict_proba(observation)[0]
    predicted_state = str(model.predict(observation).ravel()[0])
    probability_table = pd.DataFrame(
        {"Probabilité": probabilities}, index=model.classes_
    ).sort_values("Probabilité", ascending=False)

    result, chart = st.columns([1, 2])
    result.metric("État prédit", predicted_state, f"Confiance : {probabilities.max():.1%}")
    result.warning("Résultat indicatif : valider toute intervention avec les procédures de sécurité du site.")
    chart.bar_chart(probability_table)

    with st.expander("Aperçu des données utilisées"):
        st.dataframe(dataframe.head(20), width="stretch")
