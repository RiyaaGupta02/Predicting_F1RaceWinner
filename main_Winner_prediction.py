import pandas as pd
import joblib
import streamlit as st
import os
from sklearn.preprocessing import StandardScaler

st.title('Predicting the Winner of the 2025 Australian GP')
st.write("This site predicts the winner of the 2025 Australian GP based on qualifying data, using the GradientBoostingAlgorithm.")

# ✅ Step 1: Load the Trained Model (Check if Exists)
model_path = "gradient_boosting_model.pkl"

if os.path.exists(model_path):
    model_data = joblib.load(model_path)
    gbm = model_data["model"]  # Extract model
    scaler = model_data["scaler"]  # Extract scaler
    expected_features = model_data["features"]  # Extract saved feature names
    st.success("✅ Model Loaded Successfully!")
else:
    st.error("🚨 Model file not found! Please train the model first.")
    st.stop()

# ✅ Step 2: Load Data
qualifying_data = pd.read_csv('Qualifying_2025RaceData.csv')  # Update file name
training_data = pd.read_csv('Final_Training_Dataset.csv')  # Update file name

# ✅ Step 3: Convert Team & Driver Names into Numerical IDs
team_mapping = {
    'McLaren': 1, 'Red Bull Racing': 2, 'Mercedes': 3, 'RB': 4, 'Williams': 5,
    'Ferrari': 6, 'Alpine': 7, 'Aston Martin': 8, 'Kick Sauber': 9, 'Haas F1 Team': 10
}

driver_mapping = {
    'VER': 1, 'HAM': 2, 'LEC': 3, 'ALO': 4, 'NOR': 5, 'PIA': 6, 'RUS': 7, 'TSU': 8,
    'ALB': 9, 'GAS': 10, 'SAI': 11, 'STR': 12, 'ANT': 13, 'HUL': 14, 'LAW': 15, 'OCO': 16, 'BEA': 17
}

# Map categorical values to numerical IDs
qualifying_data["Team"] = qualifying_data["Team"].map(team_mapping)
qualifying_data["Driver"] = qualifying_data["Driver"].map(driver_mapping)

# ✅ Step 4: Merge AvgLapTime & BestSectorTime into Qualifying Data
training_data = training_data[["Driver", "AvgLapTime", "BestSectorTime"]]
qualifying_data = qualifying_data.merge(training_data, on="Driver", how="left")

# ✅ Step 5: Handle Missing Values
qualifying_data["AvgLapTime"].fillna(training_data["AvgLapTime"].mean(), inplace=True)
qualifying_data["BestSectorTime"].fillna(training_data["BestSectorTime"].mean(), inplace=True)

# ✅ Step 6: Convert Q1, Q2, Q3 to Seconds
for col in ["Q1", "Q2", "Q3"]:
    qualifying_data[col] = qualifying_data[col].apply(
        lambda x: sum(float(t) * 60**i for i, t in enumerate(reversed(str(x).split(":"))))
    )

# ✅ Step 7: Ensure Features Match Training Data
features = ["Driver", "Team", "Laps", "Q1", "Q2", "Q3", "AvgLapTime", "BestSectorTime"]
qualifying_data = qualifying_data[features]  # Keep only expected features

# ✅ Step 8: Standardize Numerical Features (Exclude "Driver" & "Team")
num_features = ["Laps", "Q1", "Q2", "Q3", "AvgLapTime", "BestSectorTime"]
qualifying_data[num_features] = scaler.transform(qualifying_data[num_features])

# ✅ Step 9: Handle Any Remaining Missing Values
if qualifying_data.isnull().values.any():
    st.warning("⚠️ Missing values detected! Filling missing times with median values.")
    qualifying_data.fillna(qualifying_data.median(), inplace=True)

# ✅ Step 10: Run Prediction on Button Click
if st.button("🏁 Predict Race Winner"):
    # Ensure Features Match Training Data
    qualifying_data = qualifying_data[expected_features]

    # Handle NaN Values Again Before Prediction
    qualifying_data.fillna(qualifying_data.median(), inplace=True)

    # Make Predictions
    predicted_positions = gbm.predict(qualifying_data)
    qualifying_data["PredictedPosition"] = predicted_positions

    # Sort by Predicted Position
    qualifying_data = qualifying_data.sort_values(by="PredictedPosition")

    # Display Results in Streamlit
    st.subheader("📊 Predicted Race Results")
    st.dataframe(qualifying_data[["Driver", "Team", "Q3", "PredictedPosition"]])

    # Display Winner
    predicted_winner = qualifying_data.iloc[0]["Driver"]
    st.success(f"🏆 Predicted Winner of 2025 Australian GP: {predicted_winner}")