import pandas as pd
import joblib
import streamlit as st
import pickle
import os
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt



from prediction1 import run_prediction

# Title and Description but styled bit more
st.markdown("<h1 style='text-align: center; color: #484c94;'>🏁 Predicting the Winner of the 2025 Australian Grand Prix 🏆</h1>", unsafe_allow_html=True)

st.sidebar.title("🏎️About this Site")
# description with some styling
st.sidebar.markdown("""
<div style='font-size:16px; line-height:1.6;'>
🚗 This tool uses <strong>real F1 qualifying data</strong> from past races and combines it with advanced machine learning algorithms like here using Gradient Boosting algorithm to simulate the outcome of the 2025 Australian Grand Prix.<br><br>
🧠 We've trained two different models — a <b>realistic lap-time based model</b> and a <b>model thats weighted on important specific features</b> — and <u>blended their predictions</u> to form an ensemble that gives higher confidence in the winner prediction.<br><br>
🏆 The goal? Accurately predict who finishes first based on speed, consistency, experience, and more!
</div>
""", unsafe_allow_html=True)


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
num_features = ["Q1", "Q2", "Q3", "Laps", "AvgLapTime", "BestSectorTime"]
qualifying_data[num_features] = scaler.transform(qualifying_data[num_features])

# ✅ Step 9: Handle Any Remaining Missing Values
# if qualifying_data.isnull().values.any():
#     st.warning("⚠️ Missing values detected! Filling missing times with median values.")
#     qualifying_data.fillna(qualifying_data.median(), inplace=True)



# ✅ Step 10: Run Prediction on Button Click
if  st.button("🏁 Predict Race Winner"):
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
    # commenting out this part of code as resolving model and its issues and later will print the final output
    # st.subheader("📊 Predicted Race Results")
    # st.dataframe(qualifying_data[["Driver", "Team", "Q3", "PredictedPosition"]])

    # Reverse mappings
    driver_id_to_abbr = {v: k for k, v in driver_mapping.items()}
    id_to_team_abbr = {v: k for k, v in team_mapping.items()}

    # driver mapping for final output to show full names
    # This mapping is used to convert driver codes back to full names for display purposes
    driver_full_names ={
    'VER': 'Max Verstappen',
    'HAM': 'Lewis Hamilton',
    'LEC': 'Charles Leclerc',
    'ALO': 'Fernando Alonso',
    'NOR': 'Lando Norris',
    'PIA': 'Oscar Piastri',
    'RUS': 'George Russell',
    'TSU': 'Yuki Tsunoda',
    'ALB': 'Alexander Albon',
    'GAS': 'Pierre Gasly',
    'SAI': 'Carlos Sainz',
    'STR': 'Lance Stroll',
    'ANT': 'Oliver Bearman',
    'HUL': 'Nico Hülkenberg',
    'LAW': 'Liam Lawson',
    'OCO': 'Esteban Ocon',
    'BEA': 'Valtteri Bottas'
}

    # Dispplay winners with full name
    winner_row = qualifying_data.iloc[0]
    driver_id = winner_row["Driver"]
    team_id = winner_row["Team"]
    
    driver_abbr = driver_id_to_abbr.get(driver_id, "Unknown")
    driver_name = driver_full_names.get(driver_abbr, driver_abbr)
    team_name = id_to_team_abbr.get(team_id, "Unknown Team")
   # st.success(f"🏆 Predicted Winner of 2025 Australian GP: {driver_name} ({team_name})")



   # above code is for the model which we trained will be resolving the issue currently facing and will be updating the code & model feature then showinh
   # for now prediction1.py the results that we get by training model in different way on less parameters will be using that here

    results_df, shap_values_positive, model_score, shap_fig = run_prediction()

    st.subheader("🔮 Predictions")
    st.dataframe(results_df[["Driver", "WinProbability", "Team", "QualifyingPosition"]])

    st.success(f"🏆 Predicted Winner of the 2025 race: {results_df.iloc[0]['Driver']}")
    st.subheader("📊 Model Evaluation")
    st.write(f"Model Accuracy: **{model_score:.3f}** (Classification Score)")

# SHAP Analysis auto displayed cos of function ---> but this is of other way when we're predicting using different way of training the model with very limited feature that way
    # ✅ Now display the SHAP figure
    st.subheader("🔍 SHAP Analysis")
    st.pyplot(shap_fig)

    with st.expander("ℹ️ What does this SHAP chart show?", expanded=True):
        st.markdown("""
    **SHAP (SHapley Additive exPlanations)** helps us understand **why** the model is making certain predictions by assigning an importance value to each feature.

    - 📊 **Left chart**: Shows the average impact (positive or negative) of each feature on the prediction of a driver finishing in the **Top 3**.
    - 🔥 **Right chart**: A heatmap showing how much each feature influenced the top 5 drivers' probability of finishing in the Top 3.
    
    This allows us to peek under the hood of the model and validate whether it's focusing on what matters — like **qualifying position**, **car strength**, or **driver skill**.
    """)
    

    
