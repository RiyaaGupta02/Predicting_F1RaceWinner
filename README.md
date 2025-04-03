#### One of ML_MiniProject is Predicting the F1 Race Winner the readme is for the following 





# 🏎️F1 Race Winner Prediction using Gradient Boosting
This project aims to predict Formula 1 race winners based on qualifying data, lap times, Drivers team, sector performance, and other race metrics using a Gradient Boosting Model (GBM).

## 🚀Project Overview
The goal is to analyze historical qualifying performances and lap times to predict the final race winner. We use Gradient Boosting Regressor (GBM) due to its strong performance on structured numerical data.

### Dataset & Preprocessing
1️⃣ Data Sources -->
- Qualifying Data (manually took from the F1 Race Site) → Contains Q1, Q2, Q3 times, driver & team info.
- Final Training Data → Includes lap times, sector times, tire usage, and more.
- Here this Final Training Data is taken up from FastF1 API as here :
- FastF1 API: Fetches lap times, race results & other all information.
- Historical F1 Results: Processed from FastF1 for training the model. 

2️⃣ Data Preprocessing Steps -->
- Mapped Categorical Features → Converted driver & team names to numerical IDs here mapping be similar for pre-processing dataset in Historical Data as well as of Qualifying one.
- Handled Missing Values → Filled missing sector times & lap times using mean imputation.
- Converted Time Formats → Transformed Q1, Q2, Q3 from MM:SS.sss to seconds.
- Feature Selection → Selected key metrics like AvgLapTime, BestSectorTime, and qualifying performance.
- Standardization → Applied StandardScaler to normalize numerical data.

## 🏗️Model Training
#### Features Used for Training -->
- Driver, Team (Categorical → Encoded)
- Q1, Q2, Q3 (Qualifying Lap Times)
- Laps (Total laps completed in qualifying)
- AvgLapTime, BestSectorTime (Historical race performance indicators)

  ### Why GBM?
  - Handles numerical & categorical data well.
  - Learns non-linear relationships between laps occurring & race positions.
  - Performs boosting, reducing error iteratively.
 
  ### 🎯 Project Ideation & Goals
- The core idea behind this project is to analyze how well a driver performs in qualifying & practice sessions to estimate their race day performance.

Key Insights We Wanted to Explore:
- Does qualifying position always translate to race success?
- How much do sector times & lap consistency matter?
- Can a machine learning model predict F1 race winners accurately?
📌 Outcome:
- Using GBM, we achieved a model that predicts final race positions based on qualifying & practice data.

### More Improvements to do?
 - Would be to try training it on other models and seeing the impact + accuracy.
  - Also it'll be including the weather & track condition to check further on its accuracy & increasing it more.
