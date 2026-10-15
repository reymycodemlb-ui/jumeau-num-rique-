# Projet 5 — Jumeau numérique des équipements miniers

Application Streamlit pour explorer un CSV de relevés, simuler les valeurs des capteurs et estimer l’état d’un équipement avec XGBoost.

## Lancer l’application

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

L’application ouvre par défaut le CSV de démonstration dans `data/`. Pour utiliser un autre jeu de données, chargez un fichier CSV depuis la barre latérale.

## Colonnes attendues

- `Equipment` : identifiant de l’équipement
- `HealthIndex`
- `EngineTemp`
- `Vibration`
- `Pressure`
- `State` : classe à prédire (`Healthy`, `Warning`, `Critical`, ou classes équivalentes)

## Fonctions

- Vue d’ensemble des relevés et états
- Simulateur « what-if » pour ajuster les mesures d’un équipement
- Estimation de classe et probabilités XGBoost
- Aide indicative à la maintenance
- Accuracy, balanced accuracy, F1, rapport de classification et matrice de confusion
- Téléchargement des prédictions de test et des métriques

## Limite importante

Le CSV fourni ne contient pas d’horodatage. L’application classe donc un relevé et ne prévoit pas une trajectoire de dégradation ou une date de panne. Sur le CSV de démonstration, l’accuracy mesurée est d’environ 34 % (contre environ 34 % pour la référence majoritaire) : les prédictions sont expérimentales et ne doivent pas guider seules une intervention réelle ou une décision de sécurité.
