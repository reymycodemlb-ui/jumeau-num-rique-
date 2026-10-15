from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


st.set_page_config(
    page_title="Jumeau numérique minier",
    page_icon="⛏️",
    layout="wide",
)

FEATURES = ["HealthIndex", "EngineTemp", "Vibration", "Pressure"]
TARGET = "State"
GROUP = "Equipment"
DEFAULT_CSV = Path(__file__).parent / "data" / "pasted_file_8Kje44_digitalTwin.csv"

st.markdown(
    """
    <style>
      .stApp { background: #ffffff; color: #18212b; }
      [data-testid="stHeader"] { background: rgba(255,255,255,0.96); }
      h1 { color: #111820; font-weight: 750; letter-spacing: -0.025em; }
      h2, h3 { color: #5b83b3; }
      [data-testid="stMetric"] {
        background: #f6f9fc; border: 1px solid #e4ebf2;
        padding: 16px 18px; border-radius: 12px;
      }
      div[data-testid="stSidebar"] { background: #f7f9fc; }
      .project-subtitle { color: #687788; font-size: 1.02rem; margin-top: -0.4rem; }
      .caveat { background: #fff7e6; border-left: 4px solid #e5a93d;
        padding: 0.8rem 1rem; border-radius: 6px; color: #57451e; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Projet 5. Jumeau numérique des équipements miniers")
st.markdown(
    '<div class="project-subtitle">Simuler les équipements · Prédire leur état · Assister la maintenance</div>',
    unsafe_allow_html=True,
)
st.markdown("### Objectifs")
obj1, obj2, obj3 = st.columns(3)
obj1.markdown("**01 — Simuler**  \nExplorer l’effet des mesures capteurs sur l’état estimé.")
obj2.markdown("**02 — Prédire**  \nClasser l’état observé avec un modèle XGBoost.")
obj3.markdown("**03 — Assister**  \nAfficher une recommandation de maintenance indicative.")

st.sidebar.title("Données du jumeau")
uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source_name = uploaded_file.name
elif DEFAULT_CSV.exists():
    df = pd.read_csv(DEFAULT_CSV)
    source_name = DEFAULT_CSV.name
else:
    st.info("Chargez un CSV contenant les colonnes d’équipement, de capteurs et d’état.")
    st.stop()

required_columns = [GROUP, *FEATURES, TARGET]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    st.error(f"Colonnes absentes du CSV : {', '.join(missing_columns)}")
    st.stop()

df = df.dropna(subset=required_columns).copy()
if df.empty:
    st.error("Le CSV ne contient aucune ligne complète pour les colonnes requises.")
    st.stop()

st.sidebar.caption(f"Source : {source_name}")
st.sidebar.caption(f"{len(df):,} relevés · {df[GROUP].nunique():,} équipements")


@st.cache_resource(show_spinner="Entraînement et évaluation de XGBoost…")
def train_and_evaluate(dataframe: pd.DataFrame):
    X_all = dataframe[FEATURES].astype(float)
    y_text = dataframe[TARGET].astype(str)
    groups = dataframe[GROUP].astype(str)

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(y_text)
    class_names = encoder.classes_

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups=groups))
    X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    parameters = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    validation_model = XGBClassifier(**parameters)
    validation_model.fit(X_train, y_train)
    predicted_encoded = validation_model.predict(X_test).astype(int)
    probabilities = validation_model.predict_proba(X_test)
    predicted_labels = encoder.inverse_transform(predicted_encoded)
    actual_labels = y_text.iloc[test_idx].to_numpy()

    metrics = {
        "Accuracy": accuracy_score(y_test, predicted_encoded),
        "Balanced accuracy": balanced_accuracy_score(y_test, predicted_encoded),
        "F1 macro": f1_score(y_test, predicted_encoded, average="macro"),
        "F1 pondéré": f1_score(y_test, predicted_encoded, average="weighted"),
    }
    report = pd.DataFrame(
        classification_report(
            actual_labels,
            predicted_labels,
            labels=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).T
    matrix = pd.DataFrame(
        confusion_matrix(actual_labels, predicted_labels, labels=class_names),
        index=class_names,
        columns=class_names,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))

    importance = pd.Series(
        validation_model.feature_importances_, index=FEATURES, name="Importance"
    ).sort_values(ascending=False)

    predictions = dataframe.iloc[test_idx][[GROUP]].copy()
    predictions["État réel"] = actual_labels
    predictions["État prédit"] = predicted_labels
    predictions["Confiance"] = probabilities.max(axis=1)
    for index, class_name in enumerate(class_names):
        predictions[f"Probabilité {class_name}"] = probabilities[:, index]

    final_model = XGBClassifier(**parameters)
    final_model.fit(X_all, y_all)
    return final_model, encoder, metrics, baseline_accuracy, report, matrix, importance, predictions


try:
    (model, encoder, metrics, baseline_accuracy, report, matrix,
     importance, test_predictions) = train_and_evaluate(df)
except ValueError as error:
    st.error(f"Impossible d’entraîner le modèle avec ce CSV : {error}")
    st.stop()

st.markdown(f"### Tableau de bord — {source_name}")
metric_columns = st.columns(4)
metric_columns[0].metric("Relevés", f"{len(df):,}")
metric_columns[1].metric("Équipements", f"{df[GROUP].nunique():,}")
metric_columns[2].metric("Accuracy test", f"{metrics['Accuracy']:.1%}")
metric_columns[3].metric("F1 macro", f"{metrics['F1 macro']:.3f}")

if metrics["Accuracy"] <= baseline_accuracy + 0.02:
    st.warning(
        f"Le modèle n’améliore pas nettement la référence majoritaire "
        f"({baseline_accuracy:.1%}). Les prédictions sont expérimentales et "
        "ne doivent pas être utilisées seules pour décider d’une intervention de sécurité."
    )

st.markdown(
    '<div class="caveat"><b>Limite des données :</b> le CSV ne contient pas d’horodatage. '
    'L’interface simule et classe un relevé, mais ne peut pas encore prévoir une évolution '
    'ou une date de panne.</div>',
    unsafe_allow_html=True,
)


tab_overview, tab_sim, tab_maintenance, tab_model, tab_data = st.tabs(
    ["Vue d’ensemble", "Simulateur", "Maintenance", "Modèle & métriques", "Données"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Répartition des états")
        state_counts = df[TARGET].value_counts().rename_axis("État").to_frame("Nombre")
        st.bar_chart(state_counts)
    with right:
        st.subheader("Température moteur et vibration")
        st.scatter_chart(df, x="EngineTemp", y="Vibration", color=TARGET)
    st.subheader("Indicateurs — moyenne par état déclaré")
    st.dataframe(df.groupby(TARGET)[FEATURES].mean().round(2), width="stretch")

with tab_sim:
    st.subheader("Simuler un relevé")
    equipment_values = sorted(df[GROUP].astype(str).unique())
    selected_equipment = st.selectbox("Équipement", equipment_values)
    equipment_rows = df[df[GROUP].astype(str) == selected_equipment].reset_index(drop=True)
    record_index = st.selectbox(
        "Relevé à utiliser comme point de départ (pas d’horodatage disponible)",
        options=list(range(len(equipment_rows))),
        format_func=lambda index: f"Relevé {index + 1}",
    )
    base_row = equipment_rows.iloc[record_index]

    st.caption("Ajustez les mesures pour effectuer une simulation de type ‘what-if’.")
    columns = st.columns(4)
    simulated_values = {}
    for column, feature in zip(columns, FEATURES):
        minimum = float(df[feature].min())
        maximum = float(df[feature].max())
        simulated_values[feature] = column.number_input(
            feature,
            min_value=minimum,
            max_value=maximum,
            value=float(base_row[feature]),
            step=max((maximum - minimum) / 100, 0.01),
            format="%.2f",
            key=f"{selected_equipment}_{record_index}_{feature}",
        )

    simulation = pd.DataFrame([simulated_values], columns=FEATURES)
    simulation_probabilities = model.predict_proba(simulation)[0]
    simulated_label = encoder.inverse_transform([int(np.argmax(simulation_probabilities))])[0]
    confidence = float(np.max(simulation_probabilities))

    result_col, probability_col = st.columns([1, 2])
    with result_col:
        st.metric("État estimé", str(simulated_label), f"Confiance {confidence:.1%}")
        st.caption("Estimation du classifieur — pas un diagnostic certifié.")
    with probability_col:
        probability_frame = pd.DataFrame(
            {"Probabilité": simulation_probabilities}, index=encoder.classes_
        )
        st.bar_chart(probability_frame)

with tab_maintenance:
    st.subheader("Assistant de maintenance")
    st.write("Sélectionnez un équipement dans l’onglet **Simulateur** pour tester un relevé.")
    recommendations = {
        "Healthy": ("Surveillance normale", "Poursuivre les inspections planifiées et conserver les relevés."),
        "Warning": ("Inspection à planifier", "Vérifier les capteurs et planifier un contrôle technique selon les procédures du site."),
        "Critical": ("Vérification prioritaire", "Faire confirmer l’alerte par un technicien et appliquer les procédures de sécurité du site."),
    }
    st.info(
        "Les actions proposées sont indicatives. Elles ne remplacent ni les alarmes certifiées, "
        "ni les consignes d’exploitation et de sécurité du site."
    )
    current_action = recommendations.get(
        str(simulated_label),
        ("État non reconnu", "Faire vérifier les mesures et les règles de classification."),
    )
    st.markdown(f"### État simulé : {simulated_label}")
    st.markdown(f"**{current_action[0]}** — {current_action[1]}")
    st.markdown("**Logique de recommandation**")
    for state, (title, message) in recommendations.items():
        st.markdown(f"- **{state} — {title}** : {message}")

with tab_model:
    st.subheader("Évaluation sur des équipements non vus à l’entraînement")
    st.caption(f"Référence ‘classe majoritaire’ — accuracy : {baseline_accuracy:.1%}")
    cards = st.columns(4)
    for card, (name, value) in zip(cards, metrics.items()):
        card.metric(name, f"{value:.3f}")
    report_col, matrix_col = st.columns(2)
    with report_col:
        st.markdown("**Précision, rappel et F1 par état**")
        st.dataframe(report.round(3), width="stretch")
    with matrix_col:
        st.markdown("**Matrice de confusion**")
        st.dataframe(matrix, width="stretch")
    st.markdown("**Importance relative des variables**")
    st.bar_chart(importance)
    st.download_button(
        "Télécharger les prédictions du jeu de test",
        data=test_predictions.to_csv(index=False).encode("utf-8-sig"),
        file_name="digital_twin_test_predictions.csv",
        mime="text/csv",
    )

with tab_data:
    st.subheader("Données chargées")
    st.dataframe(df.head(100), width="stretch")
    st.caption("Aperçu limité aux 100 premières lignes.")
    st.download_button(
        "Télécharger les métriques (JSON)",
        data=pd.Series(metrics).to_json(indent=2).encode("utf-8"),
        file_name="digital_twin_metrics.json",
        mime="application/json",
    )

st.caption("Projet 5 · Jumeau numérique des équipements miniers · Modèle XGBoost")
