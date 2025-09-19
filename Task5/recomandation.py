# import des librairies
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from flask import Flask, render_template, request

# Charger les données nettoyées
df = pd.read_csv("movieclean1.csv")  # contient user_id, movie_id, rating, title, genres_list, age, gender, occupation

# Créer un encodage binaire des genres
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres_list'].apply(lambda x: x.split('|')))
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Ajouter l'encodage des genres au dataframe original
df = pd.concat([df, genres_df], axis=1)

# Créer une matrice user-movie basée sur les ratings
user_movie_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
# Calculer la similarité cosine entre les utilisateurs
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Fonction pour recommander des films pour un utilisateur
def recommend_movies(user_id, top_n=5):
    # Récupérer les utilisateurs similaires
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]  # exclure l'utilisateur lui-même
    recommendations = {}

    # Parcourir les utilisateurs similaires
    for other_user, similarity in similar_users.items():
        # Parcourir leurs films notés
        for movie_id, rating in user_movie_matrix.loc[other_user].items():
            if user_movie_matrix.loc[user_id, movie_id] == 0:  # si l'utilisateur n'a pas vu le film
                if movie_id not in recommendations:
                    recommendations[movie_id] = rating * similarity
                else:
                    recommendations[movie_id] += rating * similarity

    # Trier et retourner top N films
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [df[df['movie_id'] == movie_id]['title'].values[0] for movie_id, score in recommended_movies]
app = Flask(__name__)

# Page principale
@app.route('/')
def home():
    return render_template('index.html')

# Page de recommandation
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = recommend_movies(user_id)
    return render_template('recomand.html', user_id=user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)