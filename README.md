# 🎵 TrackNet: Machine Learning for Predicting Popular Music Tracks on Spotify

## 📝 Objective
TrackNet is a machine learning project designed to predict whether a song will become a hit or not on Spotify. By analyzing various audio and metadata features, the model identifies patterns that contribute to a song's success.

## 📂 Dataset
The dataset contains various attributes of songs, including:
- 🎤 **Artist Name** - Name of the performing artist
- 🎵 **Track Name** - Title of the song
- 🎼 **Danceability** - A measure of how suitable a track is for dancing
- 🎹 **Energy** - A measure of intensity and activity in the track
- 🥁 **Instrumentalness** - The probability of a song being instrumental
- 🎤 **Loudness** - The overall loudness of the track in decibels (dB)
- ⏱️ **Tempo** - Beats per minute (BPM) of the track
- 🎶 **Valence** - A measure of the musical positiveness of a track
- 🔄 **Key** - The key in which the song is played
- 🎵 **Mode** - Indicates major or minor scale
- 🔊 **Speechiness** - The presence of spoken words in a track
- ✅ **Hit/Not Hit (Target Variable)** - 1 (Hit Song) / 0 (Not a Hit)

## 🚀 Steps Involved
1️⃣ **Data Collection & Preprocessing:**
   - Load and explore the dataset
   - Handle missing values and outliers
   - Encode categorical variables (if applicable)
   - Normalize numerical features for better model performance

2️⃣ **Data Splitting:**
   - Divide the dataset into training and testing sets
   - Ensure class balance for accurate predictions

3️⃣ **Model Building & Training:**
   - Train multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, etc.)
   - Evaluate performance using accuracy, precision, recall, and F1-score

4️⃣ **Model Selection & Evaluation:**
   - Choose the best-performing model based on metrics
   - Save the trained model as `.pkl` for future predictions

5️⃣ **Manual Testing:**
   - Load the saved model
   - Preprocess and make predictions on new song data

6️⃣ **API Integration (Flask Web App):**
   - Build a user-friendly web interface using Flask
   - Allow users to input song attributes and get hit predictions

## 🛠️ Technologies Used
- 🐍 **Python**
- 🎵 **Spotify API (if applicable)**
- 🤖 **Scikit-learn**
- 🔢 **Pandas & NumPy**
- 📊 **Matplotlib & Seaborn** (for visualization)
- 🌐 **Flask (for web app integration)**
- 🏗️ **Machine Learning Algorithms**

## ⚙️ Installation & Setup
### Prerequisites
Ensure Python is installed and install required dependencies:
```sh
pip install -r requirements.txt
```

### ▶️ Running the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/TrackNet-Spotify-Prediction.git
   cd TrackNet-Spotify-Prediction
   ```
2. Run the Jupyter Notebook for model training:
   ```sh
   jupyter notebook "Spotify.ipynb"
   ```
3. Run the Flask web app (if applicable):
   ```sh
   python app.py
   ```

