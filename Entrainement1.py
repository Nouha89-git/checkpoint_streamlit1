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

# Fonction principale pour le traitement des données et l'entraînement du modèle
def prepare_and_train():
    # Chargement des données
    df = pd.read_csv('expresso_churn_model.csv')

def preprocess_data(df):
    # Suppression des doublons
    df = df.drop_duplicates()

    # Encodage des variables catégorielles
    for col in ['REGION', 'TOP_PACK']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Séparation des caractéristiques et de la cible
    X = df.drop(['CHURN', 'user_id'], axis=1, errors='ignore')
    y = df['CHURN']

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, scaler

# Fonction pour créer et entraîner le modèle
def train_model(X_scaled, y):
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Entraîner un modèle simple
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model