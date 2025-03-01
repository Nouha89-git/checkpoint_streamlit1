# 2. Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import streamlit as st

# Application Streamlit
def create_streamlit_app():
    st.title("Prédiction de Désabonnement - Expresso")
    st.write("Cette application prédit la probabilité de désabonnement d'un client Expresso.")

    try:
        # Chargement du modèle et des transformateurs
        with open('expresso_churn_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('expresso_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('expresso_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)

        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        # Création du formulaire de saisie
        st.header("Informations du client")

        # Dictionnaire pour stocker les valeurs d'entrée
        input_data = {}

        # Créez les champs de saisie en fonction des caractéristiques
        col1, col2 = st.columns(2)

        with col1:
            # Variables catégorielles
            if 'REGION' in feature_names:
                region_options = list(encoders['REGION'].classes_)
                selected_region = st.selectbox("Région", region_options)
                input_data['REGION'] = encoders['REGION'].transform([selected_region])[0]

            if 'TOP_PACK' in feature_names:
                top_pack_options = list(encoders['TOP_PACK'].classes_)
                selected_top_pack = st.selectbox("Top Pack", top_pack_options)
                input_data['TOP_PACK'] = encoders['TOP_PACK'].transform([selected_top_pack])[0]

            # Variables numériques (première moitié)
            if 'TENURE' in feature_names:
                input_data['TENURE'] = st.number_input("Durée dans le réseau (jours)", min_value=0)

            if 'MONTANT' in feature_names:
                input_data['MONTANT'] = st.number_input("Montant de recharge", min_value=0.0)

            if 'FREQUENCE_RECH' in feature_names:
                input_data['FREQUENCE_RECH'] = st.number_input("Fréquence de recharge", min_value=0)

            if 'REVENUE' in feature_names:
                input_data['REVENUE'] = st.number_input("Revenu mensuel", min_value=0.0)

            if 'ARPU_SEGMENT' in feature_names:
                input_data['ARPU_SEGMENT'] = st.number_input("ARPU Segment", min_value=0.0)

        with col2:
            # Variables numériques (deuxième moitié)
            if 'FREQUENCE' in feature_names:
                input_data['FREQUENCE'] = st.number_input("Fréquence", min_value=0)

            if 'DATA_VOLUME' in feature_names:
                input_data['DATA_VOLUME'] = st.number_input("Volume de données", min_value=0.0)

            if 'ON_NET' in feature_names:
                input_data['ON_NET'] = st.number_input("Appels inter Expresso", min_value=0.0)

            if 'ORANGE' in feature_names:
                input_data['ORANGE'] = st.number_input("Appels vers Orange", min_value=0.0)

            if 'TIGO' in feature_names:
                input_data['TIGO'] = st.number_input("Appels vers Tigo", min_value=0.0)

            if 'ZONE1' in feature_names:
                input_data['ZONE1'] = st.number_input("Appels vers Zone 1", min_value=0.0)

            if 'ZONE2' in feature_names:
                input_data['ZONE2'] = st.number_input("Appels vers Zone 2", min_value=0.0)

        # Troisième section pour les caractéristiques restantes
        if 'MRG' in feature_names:
            input_data['MRG'] = st.number_input("MRG (client qui fait du VAS)", min_value=0, max_value=1)

        if 'REGULARITY' in feature_names:
            input_data['REGULARITY'] = st.number_input("Régularité (jours actifs sur 90 jours)", min_value=0,
                                                       max_value=90)

        if 'FREQ_TOP_PACK' in feature_names:
            input_data['FREQ_TOP_PACK'] = st.number_input("Fréquence d'activation des Top Packs", min_value=0)

        # Bouton de prédiction
        predict_button = st.button("Prédire le désabonnement")

        if predict_button:
            # Préparer les données d'entrée
            input_df = pd.DataFrame([input_data])

            # S'assurer que toutes les colonnes nécessaires sont présentes
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Réorganiser les colonnes pour correspondre à l'ordre d'entraînement
            input_df = input_df[feature_names]

            # Standardisation des données
            input_scaled = scaler.transform(input_df)

            # Prédiction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            # Affichage du résultat
            st.header("Résultat de la prédiction")

            if prediction == 1:
                st.error("⚠️ Risque élevé de désabonnement! ⚠️")
                st.write(f"Probabilité de désabonnement: {prediction_proba[1]:.2%}")
            else:
                st.success("✅ Client fidèle avec faible risque de désabonnement")
                st.write(f"Probabilité de fidélité: {prediction_proba[0]:.2%}")

            # Affichage de l'importance des caractéristiques
            st.subheader("Facteurs d'influence")

            # Calculer l'importance des caractéristiques pour cette prédiction spécifique
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Value': input_df.values[0],
                'Importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False).head(5)

            # Créer un graphique à barres pour l'importance des caractéristiques
            st.bar_chart(feature_importance.set_index('Feature')['Importance'])

            # Suggestions pour réduire le risque de désabonnement
            if prediction == 1:
                st.subheader("Suggestions pour réduire le risque de désabonnement")

                suggestions = [
                    "Offrir des promotions personnalisées basées sur les habitudes d'utilisation",
                    "Améliorer le service client pour résoudre rapidement les problèmes",
                    "Proposer des forfaits fidélité avec des avantages croissants",
                    "Contacter le client pour comprendre ses besoins",
                    "Offrir des options de forfaits plus flexibles"
                ]

                for suggestion in suggestions:
                    st.write(f"• {suggestion}")

    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle ou de la prédiction: {e}")
        st.info("Veuillez d'abord exécuter le script principal pour préparer les données et entraîner le modèle.")


# Point d'entrée principal
if __name__ == "__main__":
    # Pour l'entraînement du modèle et la préparation des données (à exécuter une seule fois)
    # model, scaler, encoders, feature_names = prepare_and_train()

    # Pour lancer l'application Streamlit (à exécuter séparément)
    # Exécutez: streamlit run ce_script.py
    create_streamlit_app()