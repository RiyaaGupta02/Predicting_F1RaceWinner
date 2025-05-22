import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def run_prediction():
    # USE: Historical performance, qualifying, team strength & manually trying to add 2025 data 
    
    # 2025 Qualifying Data
    qualifying_2025 = pd.DataFrame({
        "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", "Yuki Tsunoda",
                   "Alexander Albon", "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", "Carlos Sainz",
                   "Fernando Alonso", "Lance Stroll"],
        "QualifyingTime (s)": [75.096, 75.180, 75.481, 75.546, 75.670,
                               75.737, 75.755, 75.973, 75.980, 76.062, 76.4, 76.5],
        "QualifyingPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Grid position
        "Team": ["McLaren", "McLaren", "Red Bull", "Mercedes", "RB",
                 "Williams", "Ferrari", "Ferrari", "Alpine", "Williams", "Aston Martin", "Aston Martin"]
    })
    
    # Driver skill ratings (based on career performance, not current points)
    driver_skill_map = {
        "Lando Norris": 92,      # 2024 strong season, consistent pace
        "Oscar Piastri": 87,     # Improving rapidly, race winner
        "Max Verstappen": 98,    # Still the benchmark
        "George Russell": 84,    # Solid but not elite
        "Charles Leclerc": 91,   # Qualifying master, sometimes inconsistent
        "Lewis Hamilton": 89,    # Age factor but still elite
        "Alexander Albon": 79,   # Solid midfield
        "Carlos Sainz": 86,      # Consistent performer
        "Yuki Tsunoda": 78,      # Improving but inconsistent
        "Pierre Gasly": 81,      # Experienced
        "Fernando Alonso": 90,   # Still elite despite age
        "Lance Stroll": 74       # Consistent but limited
    }
    
    # Team car performance (2025 estimated)
    # this data is estimated based on 2024 & 2024 team performance and team changes + also synthetic data made using current track records and via helps of Gpt & other models brought
    team_performance_map = {
        "McLaren": 94,       # Dominant car
        "Red Bull": 85,      # Still strong but not dominant
        "Mercedes": 80,      # Improved from 2024
        "Ferrari": 83,       
        "Williams": 68,      
        "RB": 72,           # Decent B-team
        "Alpine": 70,        
        "Aston Martin": 66   # Fallen back
    }
    
    # Team reliability factor (impacts DNF probability)
    team_reliability_map = {
        "McLaren": 0.95,
        "Red Bull": 0.92,
        "Mercedes": 0.90,
        "Ferrari": 0.85,  # Historically less reliable
        "Williams": 0.88,
        "RB": 0.87,
        "Alpine": 0.82,
        "Aston Martin": 0.86
    }
    
    # could add track factor but it will make it generic to paticular track
    # but here adding Track factor which later can change based on the track where race is being held in world
    # Track factor (hypothetical & also based on historical data)
    # 🔧 NEW: Track-specific factors for Australia
    australia_track_factors = {
        "McLaren": 1.03,     # Good on street circuits
        "Red Bull": 0.98,    # Less dominant on street circuits
        "Mercedes": 1.01,    # Improved setup
        "Ferrari": 0.95,     
        "Williams": 1.00,    # Neutral
        "RB": 1.00,
        "Alpine": 0.98,
        "Aston Martin": 0.97
    }


    # Add base features for qualifying data
    qualifying_2025["DriverSkill"] = qualifying_2025["Driver"].map(driver_skill_map)
    qualifying_2025["TeamPerformance"] = qualifying_2025["Team"].map(team_performance_map)
    qualifying_2025["TeamReliability"] = qualifying_2025["Team"].map(team_reliability_map)
    qualifying_2025["TrackFactor"] = qualifying_2025["Team"].map(australia_track_factors)
    
    # feature to add one more for more fair & specific prediction be
    qualifying_2025["DriverCarStrength"] = (qualifying_2025["DriverSkill"] + qualifying_2025["TeamPerformance"])/ 2
    
    # other feature to add --> Top Team Indicator its --> var changes base on year & season to keep strong teams at front
    top_teams = ["McLaren", "Red Bull", "Mercedes", "Ferrari"]
    qualifying_2025["IsTopTeam"] = qualifying_2025["Team"].isin(top_teams).astype(int)
    

    # Create qualifying advantage (how much better than expected grid position)
    expected_grid = qualifying_2025["DriverSkill"] + qualifying_2025["TeamPerformance"]
    qualifying_2025["QualifyingAdvantage"] = expected_grid.rank(ascending=False) - qualifying_2025["QualifyingPosition"]
    
    # Historical training data (create synthetic but realistic data)
    # This would normally come from previous seasons
    np.random.seed(42)
    n_samples = 200  # 200 historical race results
    
    training_data = pd.DataFrame({
        "QualifyingPosition": np.random.randint(1, 21, n_samples),
        "DriverSkill": np.random.uniform(70, 98, n_samples),
        "TeamPerformance": np.random.uniform(60, 95, n_samples),
        "TeamReliability": np.random.uniform(0.75, 0.95, n_samples),
        "QualifyingAdvantage": np.random.uniform(-5, 5, n_samples)
    })


    # to have it more specific clear & realistic as not using lot of data adding new features
    # Add new features to training data
    training_data["TrackFactor"] = np.random.uniform(0.96, 1.04, n_samples)  # Random track factors for training
    training_data["DriverCarStrength"] = (training_data["DriverSkill"] + training_data["TeamPerformance"]) / 2
    training_data["IsTopTeam"] = (training_data["TeamPerformance"] > 78).astype(int)  # Top teams have >78 performance
    
    
    # Create realistic race positions based on features
    def create_race_position(row):
        base_pos = row["QualifyingPosition"]
        skill_factor = (row["DriverSkill"] - 80) / 20  # Normalize around 80
        team_factor = (row["TeamPerformance"] - 75) / 25  # Normalize around 75
        qual_factor = row["QualifyingAdvantage"] / 10
        strength_factor = (row["DriverCarStrength"] - 77.5) / 15  # Combined strength
        top_team_bonus = -0.5 if row["IsTopTeam"] else 0  # Small advantage for top teams
        
        reliability_penalty = 0 if np.random.random() < row["TeamReliability"] else 15

        
        # Weighted impact
        position_change = (skill_factor * 2.5 + team_factor * 3.5 + qual_factor * 2 + 
                          strength_factor * 1.5 + top_team_bonus) * -2
        final_position = max(1, min(20, base_pos + position_change + reliability_penalty + np.random.normal(0, 1.5)))
        return int(final_position)
    
    training_data["RacePosition"] = training_data.apply(create_race_position, axis=1)
    
    # ✅ CORRECT: Predict top 3 finish probability instead of exact position
    training_data["TopFinish"] = (training_data["RacePosition"] <= 3).astype(int)
    
    # Train the model
    feature_cols = ["QualifyingPosition", "DriverSkill", "TeamPerformance", 
                   "TeamReliability", "QualifyingAdvantage", "DriverCarStrength", "IsTopTeam"]
    
    X_train = training_data[feature_cols]
    y_train = training_data["TopFinish"]
    
    # ✅ CORRECT: Apply custom weights by modifying feature values BEFORE scaling
    # But scale appropriately to maintain relationships
    X_weighted = X_train.copy()
    
    # Apply feature importance weights ---> super imp for model so that its able to focus ur custom based on feature importance
    # These weights are hypothetical and should be adjusted based on domain knowledge
    weights = {
        "QualifyingPosition": 3.0,      # Qualifying is very important
        "DriverSkill": 2.0,             # Driver skill matters
        "TeamPerformance": 2.5,         # Car performance crucial
        "TeamReliability": 1.5,         
        "QualifyingAdvantage": 1.8,     
        "DriverCarStrength": 2.2,
        "IsTopTeam": 1.3          
    }
    
    # apply custom weights by modifying feature values before scaling
    X_weighted = X_train.copy()

    # Scale features individually to preserve importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_weighted)
    
    # Apply weights after scaling
    for i, col in enumerate(feature_cols):
        if col in weights:  # Safety check
            X_scaled[:, i] *= weights[col]
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_scaled, y_train)
    
    # Predict for 2025 Australian GP
    X_2025 = qualifying_2025[feature_cols]
    X_2025_weighted = X_2025.copy()
    X_2025_scaled = scaler.transform(X_2025_weighted)
    
    # Apply same weights
    for i, col in enumerate(feature_cols):
        if col in weights:  # Safety check
            X_2025_scaled[:, i] *= weights[col]
    
    # Get win probabilities
    win_probabilities = model.predict_proba(X_2025_scaled)[:, 1]
    qualifying_2025["WinProbability"] = win_probabilities
    
    # Sort by win probability
    results = qualifying_2025.sort_values("WinProbability", ascending=False)
    
    # Calculate model score
    model_score = model.score(X_scaled, y_train)
    
    
    print("🏁 2025 Australian GP Predictions:")
    print("=" * 50)
    for i, (_, row) in enumerate(results.head(5).iterrows()):
        print(f"{i+1:2d}. {row['Driver']:15s} - {row['WinProbability']:.3f} ({row['Team']})")
    

    print(f"\n📊 Model Performance: {model_score:.3f}")

          # SHAP Analysis AND PLOTTING
     # SHAP Explanation - Clean implementation
    explainer = shap.Explainer(model.predict_proba, X_scaled[:100])  # Sample for speed
    shap_values_proba = explainer(X_2025_scaled)
    
    # For binary classification, take positive class (index 1)
    shap_values_positive = shap_values_proba[:, :, 1]  # Top 3 finish probability
    
    print(f"\n🔍 SHAP Feature Importance Analysis:")
    print("=" * 50)
    
    # can't do plt.show() in streamlit--> wont work so doing modifications below ---?

    # Calculate mean absolute SHAP values for feature importance
    feature_importance = np.abs(shap_values_positive.values).mean(0)
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        print(f"{row['Feature']:20s}: {row['Importance']:.4f}")
    
    # Create simple SHAP visualizations - Just 2 charts
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))   # making fig size big here
    
    # 1. Overall Feature Importance Bar Chart
    ax1 = axes[0]
    bars = ax1.barh(importance_df['Feature'], importance_df['Importance'], 
                    color='#f7578c', edgecolor='red', alpha=0.8)
    ax1.set_title('🔍 SHAP Feature Importance\n(Impact on Top 3 Predictions)', 
                  fontweight='bold', fontsize=16)
    ax1.set_xlabel('Mean |SHAP Value|', fontsize=13)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, importance_df['Importance']):
        ax1.text(value + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', ha='left', fontsize=11, fontweight='bold', color='black')
    
    # 2. SHAP Heatmap for Top 5 Drivers
    ax2 = axes[1]
    top_5_drivers = results.head(5)['Driver'].values
    top_5_shap = shap_values_positive[:5].values.T  # Features x Top 5 drivers
    
    im = ax2.imshow(top_5_shap, cmap='RdBu_r', aspect='auto')
    ax2.set_title('🔥 SHAP Heatmap: Top 5 Drivers\n(Red=Negative, Blue=Positive)', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(top_5_drivers, rotation=45, ha='right')
    ax2.set_yticks(range(len(feature_cols)))
    ax2.set_yticklabels(feature_cols)
    
    # Add text annotations
    for i in range(len(feature_cols)):
        for j in range(5):
            color = "white" if abs(top_5_shap[i, j]) > 0.02 else "black"
            text = ax2.text(j, i, f'{top_5_shap[i, j]:.3f}', 
                           ha="center", va="center", color=color, fontweight='bold', fontsize=10)
    
    # Add colorbar
    # cbar = plt.colorbar(im, ax=ax2, label='SHAP Value')
    cbar = plt.colorbar(im, ax=ax2, pad=0.01)
    cbar.set_label("SHAP Value", fontsize=12)

    
    plt.tight_layout()

# ORIGINAL RETURINING NOW FUNCTION RUN_PREDICITION RESULTS
    
    return results, shap_values_positive, model_score, fig   # return the fig tool

# Run the corrected prediction
if __name__ == "__main__":
    results, shap_vals = run_prediction()