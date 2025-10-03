# Sales Forecasting Exploration and Modeling Suite

## Project Overview
This repository delivers an end-to-end workflow for exploring historical sales data, developing forecasting models, and evaluating predictive performance. The workflow is built for the provided `Sales_History_Dataset.xlsx` file (convertible to `Sales_History_Dataset.csv`) and demonstrates how exploratory analysis, classical statistics, and machine-learning techniques can be combined to understand and anticipate demand trends.

### Objectives
- Inspect the sales history to understand product performance, seasonality, and volume trends.
- Generate baseline visualizations that communicate demand patterns to business stakeholders.
- Prepare a continuous daily time series suitable for forecasting experiments.
- Train a diverse set of forecasting algorithms to compare strengths and weaknesses.
- Quantitatively evaluate every model and publish a markdown performance report.

## Installation
1. Ensure you are using Python 3.9 or later.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Convert the Excel workbook to CSV for faster downstream loading:
   ```bash
   python convert_data.py
   ```
   This creates `Sales_History_Dataset.csv`, which the analysis and forecasting scripts consume.

## How to Run the Analysis
1. **Exploratory Data Analysis** – Execute the main script to profile the dataset and generate descriptive plots:
   ```bash
   python main.py
   ```
   - Console output summarises schema information and descriptive statistics.
   - Visualizations are saved in the `outputs/` directory.

2. **Forecast Model Training** – Import the helpers in `forecasting_models.py` to create train/test splits and obtain predictions. Example usage:
   ```python
   import pandas as pd
   from forecasting_models import (
       prepare_daily_sales,
       train_test_split,
       sarima_forecast,
   )

   data = pd.read_csv("Sales_History_Dataset.csv")
   daily_series = prepare_daily_sales(data)
   train, test = train_test_split(daily_series)
   sarima_preds = sarima_forecast(train, test)
   ```

3. **Model Evaluation** – Supply the actual and predicted series to the evaluation workflow:
   ```python
   from model_evaluation import evaluate_models

   predictions = {"SARIMA": sarima_preds}
   metrics = evaluate_models(test, predictions)
   print(metrics)
   ```
   - Comparison plots, error histograms, and metric tables are written to `outputs/`.
   - `performance_report.md` is refreshed with model rankings and recommendations.

## Forecasting Algorithms
The project implements seven algorithms, enabling a balanced comparison across statistical and machine-learning approaches:
- **7-day Moving Average** – Baseline method that forecasts the next value as the mean of the most recent observations.
- **Holt-Winters Exponential Smoothing** – Captures level, trend, and seasonal components through exponentially decaying weights.
- **ARIMA (2,1,2)** – Models autocorrelation in differenced data using autoregressive and moving-average terms.
- **SARIMA (1,1,1,7)** – Extends ARIMA with weekly seasonal patterns to represent recurring demand cycles.
- **Prophet** – Decomposes the signal into trend and seasonality with automatic changepoint detection and holiday handling.
- **Random Forest Regressor** – Tree ensemble that leverages calendar attributes, lags, and rolling statistics to model nonlinear relationships.
- **Gradient Boosting Regressor** – Sequential ensemble of shallow trees that iteratively focuses on residual errors for improved accuracy.

## Interpreting the Results
- **Exploratory Plots** – Review the daily sales trend, product distribution, and quantity time series plots to identify seasonality or anomalies before modeling.
- **Metrics Table** – Focus on RMSE for model ranking; lower values indicate better predictive fit. MAE and MAPE provide additional context for average error magnitude, while R² explains variance captured.
- **Actual vs Predicted Plot** – The highlighted line indicates the best-performing model. Assess how closely the predictions track peaks and troughs.
- **Error Histograms** – Symmetric, tightly centered distributions suggest unbiased models with low variance.
- **Performance Report** – Read `performance_report.md` for narrative insights, recommended models, and next-step suggestions tailored to the observed sales patterns.

For implementation details, model rationales, and future enhancements, refer to `TECHNICAL_DOCUMENTATION.md`.
