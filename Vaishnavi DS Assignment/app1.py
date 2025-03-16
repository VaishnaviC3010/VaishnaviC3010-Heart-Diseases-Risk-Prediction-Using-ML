import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
from flask import Flask, request, jsonify, render_template

# Load dataset
df = pd.read_csv("heart.csv")

# Data Preprocessing
# Handling missing values
df.fillna(df.mean(), inplace=True)

# Encoding categorical variables
label_enc = LabelEncoder()
if 'sex' in df.columns:
    df['sex'] = label_enc.fit_transform(df['sex'])
else:
    print("Warning: 'sex' column not found!")

selected_features = ["age", "sex", "chol", "thalach"]
# Ensure these columns exist in the dataset
for feature in selected_features:
    if feature not in df.columns:
        raise KeyError(f"Error: '{feature}' column not found in dataset!")

X = df[selected_features]  # Features
y = df["target"]  # Target Variable

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Model Training
# Logistic Regression Model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Model Evaluation
log_reg_preds = log_reg_model.predict(X_test)
dt_preds = dt_model.predict(X_test)

print("Logistic Regression Model")
print(classification_report(y_test, log_reg_preds))
print("Decision Tree Model")
print(classification_report(y_test, dt_preds))

# Save models
pickle.dump(log_reg_model, open("logistic_regression.pkl", "wb"))
pickle.dump(dt_model, open("decision_tree.pkl", "wb"))

# Flask App
app = Flask(__name__)

# Check if model files exist
if os.path.exists("scaler.pkl") and os.path.exists("decision_tree.pkl"):
    scaler = pickle.load(open("scaler.pkl", "rb"))
    dt_model = pickle.load(open("decision_tree.pkl", "rb"))
else:
    raise FileNotFoundError("Error: Required model files not found! Train and save models first.")

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect only 4 required features
        features = [float(request.form.get(key)) for key in ["age", "sex", "chol", "thalach"]]

        # Convert to NumPy array and reshape
        input_data = np.array(features).reshape(1, -1)

        # Apply Standard Scaling
        input_data = scaler.transform(input_data)

        # Make Prediction
        prediction = dt_model.predict(input_data)
        
        return render_template('index1.html', prediction_text=f'Predicted Disease Status: {prediction[0]}')

    except Exception as e:
        return render_template('index1.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
