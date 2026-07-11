import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset and prepare
file_path = r"C:\Users\HP\Desktop\SOIL_TESTING_KIT\soil_django\analysis\data\soil_data.csv"  # Update this path
data = pd.read_csv(file_path)

# Label encoding for crop type
label_encoder_crop = LabelEncoder()
data['Crop Type Encoded'] = label_encoder_crop.fit_transform(data['Crop Type'])

# Label encoding for soil type
label_encoder_soil = LabelEncoder()
data['Soil Type Encoded'] = label_encoder_soil.fit_transform(data['Soil Type'])

# Prepare features (including encoded soil type, NPK, salinity, and other parameters)
X_crop = data[['Electrical Conductivity (dS/m)', 'Moisture (%)', 'pH', 'Temperature (°C)', 
               'Salinity (ppm)', 'Humidity (%)', 'Nitrogen (mg/kg)', 'Phosphorus (mg/kg)', 
               'Potassium (mg/kg)', 'Soil Type Encoded']]

# Target variable: Crop type
y_crop = data['Crop Type Encoded']

# Train Random Forest model for crop recommendation
rf_crop_recommendation = RandomForestClassifier(n_estimators=100, random_state=42)
rf_crop_recommendation.fit(X_crop, y_crop)

# Function to get crop recommendation
def get_crop_recommendation(user_input, predicted_soil_type, npk, salinity):
    """
    Predict crop based on user input, predicted soil type, NPK values, and salinity.
    
    :param user_input: Dictionary with user input values for crop prediction
    :param predicted_soil_type: Predicted soil type from soil analysis
    :param npk: Dictionary containing predicted nitrogen, phosphorus, and potassium values
    :param salinity: Predicted salinity value
    :return: Predicted crop
    """
    # Prepare user input for prediction, including soil type and NPK values
    user_input['Soil Type'] = predicted_soil_type
    user_input['Nitrogen (mg/kg)'] = npk['Nitrogen']
    user_input['Phosphorus (mg/kg)'] = npk['Phosphorus']
    user_input['Potassium (mg/kg)'] = npk['Potassium']
    user_input['Salinity (ppm)'] = salinity

    # Convert user input to DataFrame for prediction
    user_df = pd.DataFrame([user_input])

    # Encode the soil type for user input
    user_df['Soil Type Encoded'] = label_encoder_soil.transform(user_df['Soil Type'])

    # Ensure the columns match the training features
    required_columns = ['Electrical Conductivity (dS/m)', 'Moisture (%)', 'pH', 'Temperature (°C)', 
                        'Salinity (ppm)', 'Humidity (%)', 'Nitrogen (mg/kg)', 'Phosphorus (mg/kg)', 
                        'Potassium (mg/kg)', 'Soil Type Encoded']
    
    # Ensure the input data has all required columns in the same order as training data
    user_df = user_df[required_columns]

    # Predict crop type (encoded)
    predicted_crop_encoded = rf_crop_recommendation.predict(user_df)[0]
    
    # Decode the crop type to its original label
    predicted_crop = label_encoder_crop.inverse_transform([predicted_crop_encoded])[0]

    return predicted_crop

# Function to calculate accuracy of crop recommendation model
def get_crop_accuracy():
    """Returns the accuracy of the crop recommendation model."""
    y_pred_test = rf_crop_recommendation.predict(X_crop)
    crop_accuracy = (y_pred_test == y_crop).mean()
    return crop_accuracy