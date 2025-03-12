# Flight Price Prediction

Exploring machine learning models to predict airline ticket prices based on key features such as departure time, airline, and number of stops.

**Tech Stacks:** SQL (for data extraction), Python (ML model with Random Forest and XGBoost)

## Key Results & Insights
- The best-performing model achieved 92% estimated accuracy, corresponding to an 8% Mean Absolute Percentage Error (MAPE) on test data.
- Top 3 influential features: Time Until Flight, Trip Length, Departure & Arrival Time.
- Key Insights: Flight demand—and therefore pricing—is primarily influenced by:
  - The number of tickets remaining (which decreases over time).
  - Passengers' preference for shorter travel durations.
  - Favorable departure and arrival times (e.g., avoiding late-night or very early flights).

## Objectives  
- Perform **exploratory data analysis (EDA)** to uncover insights into airline pricing strategies.  
- Build predictive models using **Random Forest, XGBoost, and other techniques**.  
- Evaluate model performance and identify key factors influencing ticket prices.  

## Project Structure  
```
flight-price/
│── notebooks/            # Jupyter notebooks for analysis & modeling
│   ├── predict-price.ipynb  # Main notebook for training & evaluation
│── data/                 # (Optional) Dataset storage
│── README.md             # Project overview
```

## Tools & Technologies  
- **SQL** – Data extraction and transformation  
- **Python (Pandas, NumPy, Matplotlib, Seaborn)** – Data processing & visualization  
- **Scikit-learn, XGBoost, Random Forest** – Machine learning modeling  

## Dataset Access  
The dataset is sourced from **[Kaggle](https://www.kaggle.com/code/mcycee/flight-price-prediction)**. To use it:  
```bash
kaggle datasets download -d mcycee/flight-price-prediction
unzip flight-price-prediction.zip -d data/
```
Or manually place the dataset inside the `data/` folder.


*Explore the full analysis in the [notebook](notebooks/predict-price.ipynb)!*  
