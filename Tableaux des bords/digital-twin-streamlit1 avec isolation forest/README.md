# Projet 5 — Jumeau numérique des équipements miniers

Application Streamlit pour explorer des relevés d’équipements, simuler les valeurs des capteurs et détecter des relevés atypiques avec **Isolation Forest**.

## Lancer sous Anaconda

Dans Anaconda Prompt, depuis ce dossier :

```bash
conda create -n jumeau-minier python=3.11 -y
conda activate jumeau-minier
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
- `State` : facultatif ; utilisé uniquement pour évaluer les anomalies (`Healthy` comparé à `Warning`/`Critical`)

## Fonctions

- Vue d’ensemble des mesures et états fournis
- Simulateur « what-if » pour modifier les capteurs
- Détection d’anomalies et score relatif au profil normal
- Taux d’anomalies configurable dans la barre latérale
- Recommandations indicatives de maintenance
- Évaluation sur des équipements réservés au test : accuracy, balanced accuracy, précision, rappel, F1, rapport et matrice de confusion
- Export CSV des scores de test

## Fonctionnement et limites

Isolation Forest n’utilise pas `State` comme variable d’entrée. Si cette colonne existe et contient `Healthy`, l’application utilise uniquement les relevés `Healthy` du jeu d’entraînement pour définir le profil de référence sain. Les étiquettes `State` servent ensuite à une évaluation indicative : `Healthy` = normal, `Warning` ou `Critical` = anomalie.

Le taux d’anomalies configuré règle le seuil de détection ; ce n’est **pas** une probabilité de panne. Le jeu de démonstration n’a pas d’horodatage et ses mesures ne semblent pas clairement distinguer les états : les alertes doivent donc être considérées comme expérimentales, et confirmées par les techniciens et les procédures de sécurité du site.
