import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset and prepare
file_path = r"C:\Users\HP\Desktop\SOIL_TESTING_KIT\soil_django\analysis\data\soil_data.csv"  # Update this path
data = pd.read_csv(file_path)

# Drop rows with missing values (if applicable)
data.dropna(inplace=True)

# Label encoding for fertilizer
label_encoder_fertilizer = LabelEncoder()
data['Fertilizer Encoded'] = label_encoder_fertilizer.fit_transform(data['Fertilizer'])

# Label encoding for soil type
label_encoder_soil = LabelEncoder()
data['Soil Type Encoded'] = label_encoder_soil.fit_transform(data['Soil Type'])

# Label encoding for crop type
label_encoder_crop = LabelEncoder()
data['Crop Type Encoded'] = label_encoder_crop.fit_transform(data['Crop Type'])

# Prepare features (including encoded soil type, NPK, salinity, and other parameters)
X_fertilizer = data[['Electrical Conductivity (dS/m)', 'Moisture (%)', 'pH', 'Temperature (°C)', 
                     'Salinity (ppm)', 'Humidity (%)', 'Nitrogen (mg/kg)', 'Phosphorus (mg/kg)', 
                     'Potassium (mg/kg)', 'Soil Type Encoded', 'Crop Type Encoded']]

# Target variable: Fertilizer
y_fertilizer = data['Fertilizer Encoded']

# Train Random Forest model for fertilizer recommendation
rf_fertilizer_recommendation = RandomForestClassifier(n_estimators=100, random_state=42)
rf_fertilizer_recommendation.fit(X_fertilizer, y_fertilizer)

# Function to get fertilizer recommendation
def recommend_fertilizer(user_input, predicted_soil_type, npk, salinity, predicted_crop):
    """
    Predict fertilizer based on user input, predicted soil type, NPK values, salinity, and predicted crop.
    
    :param user_input: Dictionary with user input values for fertilizer prediction
    :param predicted_soil_type: Predicted soil type from soil analysis
    :param npk: Dictionary containing predicted nitrogen, phosphorus, and potassium values
    :param salinity: Predicted salinity value
    :param predicted_crop: Predicted crop type (encoded)
    :return: Predicted fertilizer
    """
    # Prepare user input for prediction, including soil type, NPK values, and crop type
    user_input['Soil Type'] = predicted_soil_type
    user_input['Nitrogen (mg/kg)'] = npk['Nitrogen']
    user_input['Phosphorus (mg/kg)'] = npk['Phosphorus']
    user_input['Potassium (mg/kg)'] = npk['Potassium']
    user_input['Salinity (ppm)'] = salinity
    user_input['Crop Type'] = predicted_crop  # Include predicted crop type as a feature

    # Convert user input to DataFrame for prediction
    user_df = pd.DataFrame([user_input])

    # Encode the soil type and crop type for user input
    user_df['Soil Type Encoded'] = label_encoder_soil.transform(user_df['Soil Type'])
    user_df['Crop Type Encoded'] = label_encoder_crop.transform(user_df['Crop Type'])

    # Ensure the columns match the training features
    required_columns = ['Electrical Conductivity (dS/m)', 'Moisture (%)', 'pH', 'Temperature (°C)', 
                        'Salinity (ppm)', 'Humidity (%)', 'Nitrogen (mg/kg)', 'Phosphorus (mg/kg)', 
                        'Potassium (mg/kg)', 'Soil Type Encoded', 'Crop Type Encoded']
    
    # Ensure the input data has all required columns in the same order as training data
    user_df = user_df[required_columns]

    # Predict fertilizer type (encoded)
    predicted_fertilizer_encoded = rf_fertilizer_recommendation.predict(user_df)[0]
    
    # Decode the fertilizer type to its original label
    predicted_fertilizer = label_encoder_fertilizer.inverse_transform([predicted_fertilizer_encoded])[0]

    return predicted_fertilizer

# Function to calculate accuracy of fertilizer recommendation model
def get_fertilizer_accuracy():
    """Returns the accuracy of the fertilizer recommendation model."""
    y_pred_test = rf_fertilizer_recommendation.predict(X_fertilizer)
    fertilizer_accuracy = (y_pred_test == y_fertilizer).mean()
    return fertilizer_accuracy