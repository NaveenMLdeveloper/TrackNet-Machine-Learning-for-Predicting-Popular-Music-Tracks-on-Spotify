import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import spotify_preprocessing as sp
from sklearn.preprocessing import StandardScaler

# Set the directory to save all plots
save_path = 'plots'
os.makedirs(save_path, exist_ok=True)

# Load and explore data
file_path = "Spotify/spotifyData10.csv"
data = sp.load_and_explore_data(file_path)

# Clean data
data_cleaned = sp.clean_data(data)

# Handle outliers
data_no_outliers = sp.remove_outliers(data_cleaned)

# Scale features
data_scaled = sp.scale_features(data_no_outliers)

# Split data into features and target
X = data_scaled.drop(columns=["target"])
y = data_scaled["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

# Perform Grid Search
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)

# Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Optimized XGBoost Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Not Popular', 'Popular'], yticklabels=['Not Popular', 'Popular'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Save the Confusion Matrix plot
plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
plt.close()  # Close the plot to prevent display

# ROC Curve
y_probs = best_xgb.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Save the ROC Curve plot
plt.savefig(os.path.join(save_path, 'roc_curve.png'))
plt.close()  # Close the plot to prevent display

# Save the trained model
joblib.dump(best_xgb, "spotify_popularity_model.pkl")

# Initialize StandardScaler
scaler = StandardScaler()
# Save the scaler
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")
