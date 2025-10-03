# Technical Documentation

## Algorithm Selection Rationale
- **Moving Average (7-day window):** Provides a simple baseline that smooths noise and offers a quick benchmark for more sophisticated models.
- **Holt-Winters Exponential Smoothing:** Chosen for its ability to jointly model level, trend, and seasonality with minimal parameter tuning, ideal for weekly sales cycles.
- **ARIMA (2,1,2):** Captures autocorrelation and gradual trend changes in differenced sales series, offering interpretable coefficients for diagnostic analysis.
- **SARIMA (1,1,1,7):** Extends ARIMA with a weekly seasonal component to accommodate recurring demand spikes and dips across calendar weeks.
- **Prophet:** Selected for robustness to missing data, automatic changepoint detection, and intuitive seasonal decomposition favored by business analysts.
- **Random Forest Regressor:** Handles nonlinear relationships between engineered calendar/lag features and demand, while remaining resilient to outliers.
- **Gradient Boosting Regressor:** Excels at capturing subtle patterns by sequentially correcting residual errors, often outperforming bagging methods on structured tabular data.

## Mathematical Concepts (Plain Language)
- **Moving Average:** Calculates the forecast as the arithmetic mean of the most recent observations, effectively smoothing noise.
- **Exponential Smoothing:** Applies exponentially decaying weights to past values so recent observations influence the forecast more than older ones.
- **ARIMA:** Differencing removes trends; autoregressive terms reuse past values, and moving-average terms use past forecast errors to explain remaining structure.
- **SARIMA:** Adds seasonal differencing and seasonal autoregressive/moving-average components that repeat every seven days to model weekly cycles.
- **Prophet:** Represents trend as piecewise linear segments joined at changepoints and models seasonality with sine/cosine pairs (Fourier series) for smooth periodic effects.
- **Random Forest:** Builds many decision trees on bootstrapped samples and averages their predictions to reduce variance and capture complex feature interactions.
- **Gradient Boosting:** Sequentially fits shallow trees where each new tree predicts the previous residuals, gradually improving fit through gradient descent.

## Feature Engineering Rationale
- **Calendar Features (day of week/month/quarter):** Encode predictable demand patterns driven by weekday operations, monthly budgets, or quarterly cycles.
- **Lag Features (1, 7, 14, 30 days):** Provide the model with recent demand history and weekly/monthly echoes of demand.
- **Rolling Statistics (mean/std over 7, 14, 30 days):** Supply context about local trends and volatility, enabling tree-based models to react to demand acceleration or stabilization.
- **Gap Filling with Zeros:** Ensures the time series is continuous so statistical models can operate without missing-date bias and machine-learning features have consistent temporal spacing.

## Model Assumptions and Limitations
- **Moving Average:** Assumes near-term demand is stable; cannot anticipate trend reversals or seasonality beyond the averaging window.
- **Exponential Smoothing:** Requires reasonably consistent seasonal patterns; sudden structural breaks can degrade accuracy.
- **ARIMA/SARIMA:** Depend on stationarity after differencing; parameter choices may underfit if data has complex nonlinear dynamics.
- **Prophet:** Assumes additive seasonality by default; performance may suffer with multiplicative seasonal effects unless configured accordingly.
- **Random Forest:** Lacks native time-dependence awareness beyond engineered features; extrapolation outside observed feature ranges is limited.
- **Gradient Boosting:** Sensitive to hyperparameters and can overfit noisy signals if not regularized or cross-validated.

## Improving Prediction Quality
- Incorporate external regressors (e.g., promotions, holidays, pricing) to explain demand shifts not captured by historical sales alone.
- Evaluate additional seasonalities (monthly, yearly) and holiday calendars within Prophet and SARIMA models.
- Perform hyperparameter tuning via cross-validation or automated search for ARIMA orders, tree depths, learning rates, and ensemble sizes.
- Experiment with advanced models such as XGBoost, LightGBM, or neural networks (LSTM/Temporal Convolution) for richer temporal dynamics.
- Revisit feature engineering to include cumulative promotions, stock levels, or lead-time indicators that influence order fulfillment.
- Establish a rolling retraining schedule to refresh models with the latest sales data and maintain predictive accuracy.
