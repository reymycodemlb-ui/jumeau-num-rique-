from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit


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
    '<div class="project-subtitle">Simuler les équipements · Détecter les anomalies · Assister la maintenance</div>',
    unsafe_allow_html=True,
)
st.markdown("### Objectifs")
obj1, obj2, obj3 = st.columns(3)
obj1.markdown("**01 — Simuler**  \nExplorer l’effet des mesures capteurs sur le profil d’un équipement.")
obj2.markdown("**02 — Détecter**  \nRepérer les relevés éloignés du profil de fonctionnement sain.")
obj3.markdown("**03 — Assister**  \nFournir une alerte indicative à examiner par la maintenance.")

st.sidebar.title("Données du jumeau")
uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source_name = uploaded_file.name
elif DEFAULT_CSV.exists():
    df = pd.read_csv(DEFAULT_CSV)
    source_name = DEFAULT_CSV.name
else:
    st.info("Chargez un CSV contenant les colonnes d’équipement et de capteurs.")
    st.stop()

required_columns = [GROUP, *FEATURES]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    st.error(f"Colonnes absentes du CSV : {', '.join(missing_columns)}")
    st.stop()

df = df.dropna(subset=required_columns).copy()
if df.empty:
    st.error("Le CSV ne contient aucune ligne complète pour les colonnes requises.")
    st.stop()

for feature in FEATURES:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
df = df.dropna(subset=FEATURES).copy()
if df.empty:
    st.error("Les colonnes de capteurs doivent contenir des valeurs numériques.")
    st.stop()

contamination = st.sidebar.slider(
    "Taux d’anomalies attendu",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01,
    format="%.2f",
    help="Seuil de départ estimé pour le modèle. Ajustez-le selon les procédures métier.",
)
st.sidebar.caption(f"Source : {source_name}")
st.sidebar.caption(f"{len(df):,} relevés · {df[GROUP].nunique():,} équipements")


@st.cache_resource(show_spinner="Apprentissage du profil normal et évaluation d’Isolation Forest…")
def train_and_evaluate(dataframe: pd.DataFrame, contamination_rate: float):
    X_all = dataframe[FEATURES].astype(float)
    groups = dataframe[GROUP].astype(str)
    has_labels = TARGET in dataframe.columns
    state_text = dataframe[TARGET].astype("string").str.strip() if has_labels else None

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X_all, groups=groups))
    X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]

    # Quand l’étiquette Healthy existe, elle sert uniquement à choisir le profil
    # de référence sain. Isolation Forest n’utilise pas State comme variable.
    if has_labels:
        train_states = state_text.iloc[train_idx]
        healthy_mask = train_states.str.casefold().eq("healthy").fillna(False).to_numpy()
    else:
        healthy_mask = np.zeros(len(train_idx), dtype=bool)

    if healthy_mask.sum() >= 20:
        X_normal_train = X_train.iloc[np.flatnonzero(healthy_mask)]
        reference_description = "Relevés étiquetés Healthy du jeu d’entraînement"
    else:
        X_normal_train = X_train
        reference_description = "Tous les relevés du jeu d’entraînement (aucune classe Healthy exploitable)"

    validation_model = IsolationForest(
        n_estimators=300,
        contamination=contamination_rate,
        random_state=42,
        n_jobs=-1,
    )
    validation_model.fit(X_normal_train)

    test_predictions = validation_model.predict(X_test)  # 1 = normal, -1 = anomalie
    test_scores = validation_model.decision_function(X_test)  # plus bas = plus atypique
    metrics = None
    baseline_accuracy = None
    report = None
    matrix = None

    if has_labels:
        test_states = state_text.iloc[test_idx]
        known_mask = test_states.notna().to_numpy()
        actual_states = test_states.to_numpy()
        known_states = pd.Series(actual_states[known_mask]).astype(str).str.casefold()
        if known_mask.sum() and known_states.eq("healthy").any() and (~known_states.eq("healthy")).any():
            # Référence d’évaluation binaire : Healthy = normal, Warning/Critical = anomalie.
            y_true = (~known_states.eq("healthy")).astype(int).to_numpy()
            y_pred = (test_predictions[known_mask] == -1).astype(int)
            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
                "Précision anomalies": precision_score(y_true, y_pred, zero_division=0),
                "Rappel anomalies": recall_score(y_true, y_pred, zero_division=0),
                "F1 anomalies": f1_score(y_true, y_pred, zero_division=0),
            }
            report = pd.DataFrame(
                classification_report(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                    target_names=["Healthy / normal", "Warning-Critical / anomalie"],
                    output_dict=True,
                    zero_division=0,
                )
            ).T
            matrix = pd.DataFrame(
                confusion_matrix(y_true, y_pred, labels=[0, 1]),
                index=["Réel normal", "Réel anomalie"],
                columns=["Prédit normal", "Prédit anomalie"],
            )

            train_eval_states = state_text.iloc[train_idx].dropna().str.casefold()
            train_eval_labels = (~train_eval_states.eq("healthy")).astype(int).to_numpy()
            baseline = DummyClassifier(strategy="most_frequent")
            baseline.fit(np.zeros((len(train_eval_labels), 1)), train_eval_labels)
            baseline_accuracy = accuracy_score(
                y_true, baseline.predict(np.zeros((len(y_true), 1)))
            )

    # Modèle final : entraîné sur le profil sain disponible, sinon sur tous les relevés.
    if has_labels:
        all_healthy = state_text.str.casefold().eq("healthy").fillna(False).to_numpy()
    else:
        all_healthy = np.zeros(len(dataframe), dtype=bool)
    if all_healthy.sum() >= 20:
        X_normal_all = X_all.iloc[np.flatnonzero(all_healthy)]
    else:
        X_normal_all = X_all

    final_model = IsolationForest(
        n_estimators=300,
        contamination=contamination_rate,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_normal_all)
    training_scores = final_model.decision_function(X_normal_all)

    full_predictions = final_model.predict(X_all)
    flagged_rows = dataframe.iloc[test_idx][[GROUP]].copy()
    flagged_rows["Score de normalité"] = test_scores
    flagged_rows["Anomalie détectée"] = test_predictions == -1
    flagged_rows["État fourni"] = (
        dataframe[TARGET].iloc[test_idx].to_numpy() if has_labels else "Non fourni"
    )

    anomaly_by_state = None
    if has_labels:
        overview = dataframe[[TARGET]].copy()
        overview["Anomalie détectée"] = full_predictions == -1
        anomaly_by_state = (
            overview.groupby(TARGET, dropna=False)["Anomalie détectée"]
            .agg(taux_anomalies="mean", nombre="count")
            .sort_values("taux_anomalies", ascending=False)
        )

    return (
        final_model,
        metrics,
        baseline_accuracy,
        report,
        matrix,
        training_scores,
        reference_description,
        flagged_rows,
        anomaly_by_state,
        full_predictions,
    )


try:
    (
        model,
        metrics,
        baseline_accuracy,
        report,
        matrix,
        normal_reference_scores,
        reference_description,
        test_results,
        anomaly_by_state,
        full_predictions,
    ) = train_and_evaluate(df, contamination)
except ValueError as error:
    st.error(f"Impossible d’entraîner Isolation Forest avec ce CSV : {error}")
    st.stop()

flagged_rate = float(np.mean(full_predictions == -1))
metric_columns = st.columns(4)
metric_columns[0].metric("Relevés", f"{len(df):,}")
metric_columns[1].metric("Équipements", f"{df[GROUP].nunique():,}")
metric_columns[2].metric("Relevés signalés", f"{flagged_rate:.1%}")
metric_columns[3].metric("Taux configuré", f"{contamination:.0%}")

if metrics is not None:
    st.caption(
        f"Évaluation sur des équipements non vus — accuracy {metrics['Accuracy']:.1%}, "
        f"F1 anomalies {metrics['F1 anomalies']:.3f}, référence majoritaire {baseline_accuracy:.1%}."
    )

st.markdown(
    '<div class="caveat"><b>Interprétation :</b> Isolation Forest repère des relevés atypiques, '
    'pas des pannes certaines. Le taux configuré fixe le seuil d’alerte. Le CSV ne contient pas '
    'd’horodatage : l’application ne prévoit pas une évolution de dégradation ou une date de panne.</div>',
    unsafe_allow_html=True,
)


tab_overview, tab_sim, tab_maintenance, tab_model, tab_data = st.tabs(
    ["Vue d’ensemble", "Simulateur", "Maintenance", "Anomalies & métriques", "Données"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("État fourni dans le CSV")
        if TARGET in df.columns:
            state_counts = df[TARGET].value_counts().rename_axis("État").to_frame("Nombre")
            st.bar_chart(state_counts)
        else:
            st.info("Aucune colonne State : seule la détection non supervisée est disponible.")
    with right:
        st.subheader("Température moteur et vibration")
        if TARGET in df.columns:
            st.scatter_chart(df, x="EngineTemp", y="Vibration", color=TARGET)
        else:
            st.scatter_chart(df, x="EngineTemp", y="Vibration")
    st.subheader("Mesures moyennes par état fourni")
    if TARGET in df.columns:
        st.dataframe(df.groupby(TARGET)[FEATURES].mean().round(2), width="stretch")
    else:
        st.dataframe(df[FEATURES].describe().round(2), width="stretch")

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

    st.caption("Ajustez les capteurs pour voir si le profil simulé est considéré atypique.")
    columns = st.columns(4)
    simulated_values = {}
    for column, feature in zip(columns, FEATURES):
        minimum = float(df[feature].min())
        maximum = float(df[feature].max())
        if minimum == maximum:
            minimum -= 0.01
            maximum += 0.01
        initial = float(np.clip(float(base_row[feature]), minimum, maximum))
        simulated_values[feature] = column.number_input(
            feature,
            min_value=minimum,
            max_value=maximum,
            value=initial,
            step=max((maximum - minimum) / 100, 0.01),
            format="%.2f",
            key=f"{selected_equipment}_{record_index}_{feature}",
        )

    simulation = pd.DataFrame([simulated_values], columns=FEATURES)
    anomaly_flag = int(model.predict(simulation)[0]) == -1
    normality_score = float(model.decision_function(simulation)[0])
    sorted_reference = np.sort(normal_reference_scores)
    reference_percentile = np.searchsorted(sorted_reference, normality_score, side="right") / len(sorted_reference)
    anomaly_rarity = 100 * (1 - reference_percentile)

    result_col, score_col = st.columns([1, 2])
    with result_col:
        if anomaly_flag:
            st.error("Anomalie détectée")
        else:
            st.success("Dans le profil normal")
        st.metric("Rareté par rapport au profil normal", f"{anomaly_rarity:.1f}/100")
        st.caption("Score de rareté relatif, pas une probabilité de panne.")
        if TARGET in df.columns:
            st.caption(f"État inscrit dans ce relevé : **{base_row[TARGET]}** (non utilisé comme capteur).")
    with score_col:
        st.markdown("**Mesures simulées**")
        st.dataframe(simulation.round(2), width="stretch")
        st.caption(f"Score de normalité Isolation Forest : {normality_score:.4f} — plus bas = plus atypique.")

with tab_maintenance:
    st.subheader("Assistant de maintenance")
    st.info(
        "Une anomalie est un relevé éloigné du profil normal appris. Confirmez toujours l’alerte "
        "avec l’historique, les autres capteurs et les procédures de sécurité du site."
    )
    if anomaly_flag:
        st.markdown("### Vérification recommandée")
        st.write(
            "Contrôler le relevé et le capteur, comparer avec les inspections et consulter un technicien "
            "avant toute décision opérationnelle."
        )
    else:
        st.markdown("### Profil habituel détecté")
        st.write("Poursuivre la surveillance et les inspections planifiées.")
    st.caption(
        f"Équipement simulé : {selected_equipment} · Rareté relative : {anomaly_rarity:.1f}/100 · "
        f"Taux d’anomalies configuré : {contamination:.0%}."
    )

with tab_model:
    st.subheader("Isolation Forest — détection non supervisée")
    st.write(f"Profil de référence : **{reference_description}**.")
    st.write(
        "Les étiquettes `State` servent uniquement à mesurer la concordance sur le jeu de test "
        "(Healthy = normal ; Warning/Critical = anomalie). Elles ne sont pas utilisées par l’algorithme "
        "pour calculer le score."
    )
    if metrics is not None:
        cards = st.columns(5)
        for card, (name, value) in zip(cards, metrics.items()):
            card.metric(name, f"{value:.3f}")
        st.caption(f"Accuracy de la référence majoritaire : {baseline_accuracy:.3f}")
        report_col, matrix_col = st.columns(2)
        with report_col:
            st.markdown("**Rapport binaire de détection**")
            st.dataframe(report.round(3), width="stretch")
        with matrix_col:
            st.markdown("**Matrice de confusion — test par équipement**")
            st.dataframe(matrix, width="stretch")
        st.markdown("**Taux de relevés signalés par état inscrit**")
        if anomaly_by_state is not None:
            show_table = anomaly_by_state.copy()
            show_table["taux_anomalies"] = (show_table["taux_anomalies"] * 100).round(1).astype(str) + "%"
            st.dataframe(show_table, width="stretch")
    else:
        st.info("Aucune évaluation avec étiquettes n’est disponible. Le modèle fonctionne en détection non supervisée.")

    st.download_button(
        "Télécharger les scores du jeu de test",
        data=test_results.to_csv(index=False).encode("utf-8-sig"),
        file_name="digital_twin_isolation_forest_test.csv",
        mime="text/csv",
    )

with tab_data:
    st.subheader("Données chargées")
    st.dataframe(df.head(100), width="stretch")
    st.caption("Aperçu limité aux 100 premières lignes.")
    st.download_button(
        "Télécharger les scores du jeu de test (CSV)",
        data=test_results.to_csv(index=False).encode("utf-8-sig"),
        file_name="digital_twin_isolation_forest_test.csv",
        mime="text/csv",
    )

st.caption("Projet 5 · Jumeau numérique des équipements miniers · Isolation Forest")
