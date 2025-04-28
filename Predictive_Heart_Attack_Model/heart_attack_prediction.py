"""
Heart Attack Prediction Model
This script implements a machine learning model to predict heart attacks based on patient data.
The model uses both quantitative and qualitative data including health parameters and lifestyle choices.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_and_explore_data():
    """Load the dataset and perform initial exploration."""
    print("Loading and exploring the dataset...")
    # Load the dataset
    heart_data = pd.read_csv("heart_attack_russia.csv")
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(heart_data.info())
    print("\nFirst few rows:")
    print(heart_data.head())
    print("\nDataset shape:", heart_data.shape)
    
    return heart_data

def preprocess_data(heart_data):
    """Preprocess the data including handling missing values and encoding categorical variables."""
    print("\nPreprocessing the data...")
    
    # Step 1: Handle missing values in Alcohol_Consumption
    print("\nHandling missing values in Alcohol_Consumption:")
    print("Missing values before:", heart_data['Alcohol_Consumption'].isnull().sum())
    heart_data['Alcohol_Consumption'] = heart_data['Alcohol_Consumption'].fillna('None')
    print("Missing values after:", heart_data['Alcohol_Consumption'].isnull().sum())
    
    # Step 2: Encode categorical variables
    categorical_columns = ['Gender', 'Region', 'Exercise_Level', 'Alcohol_Consumption', 
                         'Diet', 'Occupation', 'Income_Level', 'Physical_Activity',
                         'Education_Level', 'Marital_Status', 'Urban_Rural']
    
    print("\nEncoding categorical variables...")
    label_encoders = {}
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        heart_data[column] = label_encoders[column].fit_transform(heart_data[column])
        print(f"{column} unique values after encoding:", heart_data[column].unique())
    
    # Step 3: Convert boolean columns to integers
    boolean_columns = ['Smoking', 'Diabetes', 'Family_History', 'Heart_Attack', 
                      'Angina', 'Heart_Disease_History', 'Medication', 'Obesity']
    
    print("\nConverting boolean columns to integers...")
    for column in boolean_columns:
        heart_data[column] = heart_data[column].astype(int)
        print(f"{column} values after conversion:", heart_data[column].unique())
    
    return heart_data

def prepare_features_and_target(heart_data):
    """Prepare features and target variable for model training."""
    print("\nPreparing features and target variable...")
    
    # Separate features and target
    X = heart_data.drop(['ID', 'Heart_Attack'], axis=1)
    y = heart_data['Heart_Attack']
    
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nTraining set size:", X_train.shape)
    print("Testing set size:", X_test.shape)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate different classification models."""
    print("\nTraining and evaluating models...")
    
    # Train and evaluate KNN model
    print("\nTraining KNN model...")
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    print("KNN Accuracy:", accuracy_score(y_test, knn_predictions))
    print("KNN Classification Report:")
    print(classification_report(y_test, knn_predictions))
    
    # Train and evaluate Logistic Regression model
    print("\nTraining Logistic Regression model...")
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    logreg_predictions = logreg.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_predictions))
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, logreg_predictions))

def main():
    """Main function to run the heart attack prediction model."""
    # Load and explore the data
    heart_data = load_and_explore_data()
    
    # Preprocess the data
    heart_data = preprocess_data(heart_data)
    
    # Prepare features and target
    X_train, X_test, y_train, y_test = prepare_features_and_target(heart_data)
    
    # Train and evaluate models
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main() 