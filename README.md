# 🏎️ F1 Race Winner Prediction using Gradient Boosting + SHAP + Ensemble Learning

This project aims to predict Formula 1 race winners using real-world qualifying and lap-time data, enhanced by machine learning techniques. We leverage Gradient Boosting Models (GBM), SHAP analysis for interpretability, and ensemble learning for improved prediction accuracy.
![image](https://github.com/user-attachments/assets/4895ed1e-2110-40c2-bdbf-077e9aabc706)

---

## 🚀 Project Overview

The goal is to analyze historical qualifying performances and lap times to predict the final race winner. Gradient Boosting Regressor (GBM) was selected due to its strong performance with structured data. Additionally, we applied SHAP (SHapley Additive exPlanations) for feature impact analysis and ensemble modeling to blend multiple approaches for a more robust outcome.

> 🏁 **Predicted Winner of the 2025 Australian GP: Lando Norris**

---

## 📊 Dataset & Preprocessing

### 1️⃣ Data Sources:
- **Qualifying Data:** Manually collected from F1’s official site (includes Q1, Q2, Q3, driver & team).
- **Final Training Data:** Lap times, sector info, tire usage, etc., fetched via [FastF1 API](https://theoehrly.github.io/Fast-F1/).
- **Historical Results:** Aggregated from FastF1 for training.

### 2️⃣ Preprocessing Steps:
- **Mapped Categorical Features:** Converted driver & team names into numerical IDs.
- **Imputation:** Filled missing values (sector/lap times) using mean imputation.
- **Time Conversion:** Converted qualifying times (MM:SS.sss) into seconds.
- **Feature Selection:** AvgLapTime, BestSectorTime, qualifying lap times were chosen.
- **Normalization:** StandardScaler applied to all numerical features.

---

## 🏗️ Model Training

### ✅ Features Used:
- **Driver, Team** (Categorical → Encoded)
- **Q1, Q2, Q3** (Qualifying Lap Times)
- **Laps** (Total laps completed in qualifying)
- **AvgLapTime, BestSectorTime** (Performance indicators)

### 🌟 Why Gradient Boosting?
- Handles both categorical and numerical features efficiently
- Captures non-linear relationships between lap behavior and race outcome
- Uses iterative boosting to reduce error over rounds

---

## 🤖 Ensemble Modeling

To further enhance prediction reliability, we trained:
- A **realistic lap-time-based model**
- A **feature-weighted model** based on domain-specific importance

These were blended to form an **ensemble prediction**, increasing model confidence and improving accuracy.

---

## 🔍 SHAP Analysis (Model Interpretability)

We used SHAP (SHapley Additive exPlanations) to interpret our model's predictions. SHAP helps visualize feature importance and their contribution toward predicting the race winner.

![SHAP Analysis](./assets/shap_example.png) <!-- Update path if needed -->

Key insights from SHAP:
- BestSectorTime and AvgLapTime had the highest impact.
- Q1 and Q2 times were more influential than Q3 in some predictions.
- Feature interactions between team and lap times mattered.

---

## 🎯 Project Goals & Insights

- 📌 Understand how qualifying performance impacts race results.
- 📌 Evaluate the predictive power of sector and lap times.
- 📌 Develop a model that generalizes well across different races.

> Does qualifying position always translate to race success?
> How much do sector times & lap consistency matter?
> Can ML predict F1 winners with reasonable accuracy?

---

## 🔧 Future Improvements

- Include **weather** and **track condition** data to enhance the model’s generalization.
- Extend training on additional parameters and evaluate their impact.
- Explore real-time race prediction using live telemetry data via FastF1.

---

