import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score

# Load dataset and prepare
file_path = r"C:\Users\HP\Desktop\SOIL_TESTING_KIT\soil_django\analysis\data\soil_data.csv"  # Update this path
data = pd.read_csv(file_path)

label_encoder_soil = LabelEncoder()
data['Soil Type Encoded'] = label_encoder_soil.fit_transform(data['Soil Type'])

# Prepare features and target for soil type prediction
X = data[['Electrical Conductivity (dS/m)', 'Moisture (%)', 'pH', 'Temperature (°C)', 'Humidity (%)']]
y = label_encoder_soil.transform(data['Soil Type'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model for soil type classification
rf_soil_type = RandomForestClassifier(n_estimators=100, random_state=42)
rf_soil_type.fit(X_train, y_train)

# Train Random Forest models for NPK prediction (using soil type as an additional feature)
X_npk = data[['Electrical Conductivity (dS/m)', 'Moisture (%)', 'pH', 'Temperature (°C)', 'Humidity (%)', 'Soil Type Encoded']]
rf_nitrogen = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_npk, data['Nitrogen (mg/kg)'])
rf_phosphorus = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_npk, data['Phosphorus (mg/kg)'])
rf_potassium = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_npk, data['Potassium (mg/kg)'])

# Train Random Forest model for salinity prediction
rf_salinity = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_npk, data['Salinity (ppm)'])

# Function to predict soil type
def predict_soil_type(user_input):
    user_df = pd.DataFrame([user_input])
    soil_type_encoded = rf_soil_type.predict(user_df)[0]
    return label_encoder_soil.inverse_transform([soil_type_encoded])[0]

# Function to predict NPK values
def predict_npk(user_input, soil_type):
    soil_type_encoded = label_encoder_soil.transform([soil_type])[0]
    user_input['Soil Type Encoded'] = soil_type_encoded
    user_df = pd.DataFrame([user_input])
    return {
        'Nitrogen': rf_nitrogen.predict(user_df)[0],
        'Phosphorus': rf_phosphorus.predict(user_df)[0],
        'Potassium': rf_potassium.predict(user_df)[0]
    }

# Function to calculate soil score
def calculate_soil_score(npk):
    optimal = {'Nitrogen': (30, 70), 'Phosphorus': (10, 40), 'Potassium': (40, 100)}
    score = lambda v, r: max(0, 10 - abs(v - np.mean(r)) / (r[1] - r[0]) * 10)
    total_score = sum(score(npk[n], optimal[n]) for n in optimal)
    return (total_score / 3) * 10

# Function for soil suitability message based on score
def soil_suitability_message(score):
    if score >= 80:
        return "Good: Ideal for plant growth."
    elif score >= 50:
        return "Moderate: Needs improvement."
    return "Poor: Significant amendments required."

# Calculate and store the accuracy for soil type classification
y_pred_test = rf_soil_type.predict(X_test)
soil_type_accuracy = accuracy_score(y_test, y_pred_test)

# Function to get soil type model accuracy
def get_soil_type_accuracy():
    """Returns the accuracy of the soil type classification model."""
    return soil_type_accuracy

# Function to predict salinity (ppm)
def predict_salinity(user_input, soil_type):
    soil_type_encoded = label_encoder_soil.transform([soil_type])[0]
    user_input['Soil Type Encoded'] = soil_type_encoded
    user_df = pd.DataFrame([user_input])
    return rf_salinity.predict(user_df)[0]