import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Charger modèle et scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Charger ton dataset pour analyse descriptive
df = pd.read_csv("Mall_Customers.csv")  
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_scaled = scaler.transform(X)
df["Cluster"] = model.predict(X_scaled)

st.title("🛍️ Customer Segmentation avec KMeans")

st.write("### Entrez vos données")
income = st.number_input("Revenu annuel (k$)", min_value=0, max_value=200, step=1)
score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, step=1)

if st.button("Prédire mon cluster"):
    # Prédiction
    X_new = np.array([[income, score]])
    X_new_scaled = scaler.transform(X_new)
    cluster = model.predict(X_new_scaled)[0]

    st.success(f"✅ Vous appartenez au **Cluster {cluster}**")

    # Interprétation du cluster
    cluster_profiles = df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
    desc = {
        0: "Revenus faibles, dépenses modérées.",
        1: "Revenus élevés, dépenses élevées (gros dépensiers).",
        2: "Revenus élevés, mais dépenses modérées.",
        3: "Revenus faibles, très peu dépensiers.",
        4: "Revenus moyens, dépenses modérées."
    }
    st.info(desc.get(cluster, "Profil non défini."))

    # Visualisation des clusters
    st.write("### Visualisation des clusters")
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=df["Cluster"], palette="Set1", s=50)
    plt.scatter(X_new_scaled[:,0], X_new_scaled[:,1], c="black", marker="X", s=200, label="Vous")
    plt.xlabel("Revenu annuel (scaled)")
    plt.ylabel("Spending Score (scaled)")
    plt.legend()
    st.pyplot(plt)

    # Analyse par âge et genre
    st.write("### Analyse des clusters")
    st.write("**Répartition de l’âge par cluster**")
    age_mean = df.groupby("Cluster")["Age"].mean()
    st.bar_chart(age_mean)

    st.write("**Répartition hommes/femmes par cluster**")
    genre_dist = pd.crosstab(df["Cluster"], df["Gender"], normalize="index") * 100
    st.bar_chart(genre_dist)
