from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("spotify_popularity_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature names
feature_names = [
    "Danceability", "Energy", "Key", "Loudness", "Mode", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration (ms)", "Time Signature",
    "Chorus Hit", "Sections"
]

@app.route('/')
def home():
    return render_template("index.html", feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        features = [float(request.form[feature]) for feature in feature_names]
        
        # Scale input
        scaled_features = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        result = "🎵 Popular 🎵" if prediction == 1 else "❌ Not Popular ❌"
        
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
