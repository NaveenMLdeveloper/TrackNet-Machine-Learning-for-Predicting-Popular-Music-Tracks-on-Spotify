import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Function to load and display basic dataset info
def load_and_explore_data(file_path):
    """Load dataset and display basic information."""
    data = pd.read_csv(file_path)
    print(data.info())
    print(data.head())
    return data

# Function to clean the dataset
def clean_data(data):
    """Clean dataset by dropping unnecessary columns, removing duplicates, and fixing anomalies."""
    # Drop unnecessary columns
    data = data.drop(columns=["track", "artist", "uri"])
    
    # Check for duplicates and remove them
    print("Duplicates:", data.duplicated().sum())
    data = data.drop_duplicates()
    print("Duplicates after removal:", data.duplicated().sum())
    
    # Handle time_signature anomalies (replace 0 with mode)
    print("Rows with time_signature = 0:")
    print(data[data["time_signature"] == 0])
    
    time_sig_mode = data["time_signature"].mode()[0]
    data.loc[data["time_signature"] == 0, "time_signature"] = time_sig_mode
    
    return data

# Function to handle outliers using IQR method
def remove_outliers(data, threshold=1.5):
    """Remove outliers using the IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier mask
    outlier_mask = (data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))
    print("Potential outliers per column:", outlier_mask.sum())
    
    # Remove rows with any outliers
    data_cleaned = data[~outlier_mask.any(axis=1)]
    print("Total rows before removing outliers:", len(data))
    print("Total rows after removing outliers:", len(data_cleaned))
    
    return data_cleaned

# Function to scale features
def scale_features(data, target_column="target"):
    """Scale features and return a DataFrame with scaled features and target."""
    features = data.drop(columns=[target_column])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Convert back to DataFrame
    data_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    data_scaled[target_column] = data[target_column].values
    
    return data_scaled

# Function to train and evaluate logistic regression model
def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test):
    """Train a Logistic Regression model and evaluate its performance."""
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    
    # Predict & Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model, y_pred

# Function to visualize feature importance
def plot_feature_importance(model, X, save_path):
    """Plot feature importance for logistic regression model."""
    feature_importance = abs(model.coef_[0])
    feature_names = X.columns
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance in Logistic Regression Model')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_importance.png'))
    plt.close()

# Function to plot pairplot
def plot_pairplot(data, save_path):
    """Plot pairplot to visualize relationships between features."""
    sns.pairplot(data, hue='target', plot_kws={'alpha': 0.6}, palette='Set2', markers='o')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'pairplot.png'))
    plt.close()

# Function to plot correlation heatmap
def plot_correlation_heatmap(data, save_path):
    """Plot heatmap to visualize correlations between features."""
    corr_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'correlation_heatmap.png'))
    plt.close()

# Function to plot histogram of target distribution
def plot_target_distribution(data, save_path):
    """Plot distribution of the target variable."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=data, palette='Set2')
    plt.title('Target Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'target_distribution.png'))
    plt.close()

# Main function to execute the workflow
def main():
    """Main function to execute the data processing, model training, and evaluation workflow."""
    
    # Set the directory to save all plots
    save_path = 'plots'
    os.makedirs(save_path, exist_ok=True)
    
    # Load and explore data
    file_path = "Spotify/spotifyData10.csv"
    data = load_and_explore_data(file_path)
    
    # Clean data
    data_cleaned = clean_data(data)
    
    # Handle outliers
    data_no_outliers = remove_outliers(data_cleaned)
    
    # Scale features
    data_scaled = scale_features(data_no_outliers)
    
    # Split data into features and target
    X = data_scaled.drop(columns=["target"])
    y = data_scaled["target"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model and evaluate
    model, y_pred = train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test)
    
    # Plot and save all visualizations
    plot_feature_importance(model, X, save_path)
    plot_pairplot(data_scaled, save_path)
    plot_correlation_heatmap(data_scaled, save_path)
    plot_target_distribution(data, save_path)

# Execute the main function
if __name__ == "__main__":
    main()
