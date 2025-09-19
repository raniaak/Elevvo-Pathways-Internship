from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X = np.array([[data["income"], data["score"], data["age"]]])
    X_scaled = scaler.transform(X)
    cluster = model.predict(X_scaled)[0]
    return jsonify({"cluster": int(cluster)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
