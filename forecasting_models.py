"""Sales forecasting models and utilities.

This module prepares daily sales time series data from the historical
transaction records and offers a collection of classical and machine
learning forecasting algorithms.  Each model-specific function returns
predictions for the provided test window while leaving the caller in
control of evaluation and visualization.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def prepare_daily_sales(
    data: pd.DataFrame,
    *,
    date_col: str = "Order_Date",
    quantity_col: str = "Quantity_Sold",
) -> pd.Series:
    """Aggregate order level data into a daily quantity time series.

    The function ensures that the date column is parsed as a pandas datetime,
    aggregates the requested quantity column by day, and fills any missing
    calendar dates with zeros so downstream models receive a continuous
    sequence.

    Parameters
    ----------
    data:
        Raw sales dataframe containing order level observations.
    date_col:
        Column identifying the order or invoice date. Defaults to ``"Order_Date"``.
    quantity_col:
        Column storing the numeric quantity to forecast. Defaults to
        ``"Quantity_Sold"``.

    Returns
    -------
    pandas.Series
        Daily time series indexed by ``DatetimeIndex`` with missing days filled
        by zeros.

    Raises
    ------
    KeyError
        If the ``date_col`` or ``quantity_col`` is not present in ``data``.
    """

    missing_columns = {date_col, quantity_col} - set(data.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required columns: {missing_list}")

    daily = data.copy()
    daily[date_col] = pd.to_datetime(daily[date_col], errors="coerce")
    daily = daily[[date_col, quantity_col]].dropna()
    daily = daily.sort_values(date_col)
    grouped = daily.groupby(date_col)[quantity_col].sum().sort_index()

    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    full_series = grouped.reindex(full_index, fill_value=0)
    full_series.name = quantity_col
    return full_series


def train_test_split(series: pd.Series, train_ratio: float = 0.7) -> Tuple[pd.Series, pd.Series]:
    """Split a time series into chronological training and testing partitions.

    Parameters
    ----------
    series:
        Continuous time-indexed series that should be split for modeling.
    train_ratio:
        Fraction of samples assigned to the training window. Must be between 0 and 1.

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        Two series containing the training and testing partitions, respectively.
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    ordered_series = series.sort_index()

    split_index = max(1, int(len(ordered_series) * train_ratio))
    split_index = min(split_index, len(ordered_series) - 1)
    train = ordered_series.iloc[:split_index]
    test = ordered_series.iloc[split_index:]
    return train, test


def moving_average_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    window: int = 7,
) -> pd.Series:
    """Forecast using a rolling moving average over the most recent observations.

    The moving average model is a baseline technique that predicts the next
    value as the mean of the previous ``window`` observations.  For multi-step
    forecasts, the prediction is generated sequentially by appending each
    ground-truth value to the history to preserve realism for walk-forward
    validation.

    Parameters
    ----------
    train_series:
        Historical observations used to seed the rolling window.
    test_series:
        Out-of-sample period over which predictions should be generated.
    window:
        Number of trailing observations to average for each forecasted step.

    Returns
    -------
    pandas.Series
        Forecast values indexed by the test series dates.
    """

    history = train_series.tolist()
    predictions = []
    for actual in test_series:
        lookback = history[-window:] if len(history) >= window else history
        predictions.append(float(np.mean(lookback)))
        history.append(actual)
    return pd.Series(predictions, index=test_series.index, name="moving_average")


def exponential_smoothing_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    seasonal_periods: int = 7,
) -> pd.Series:
    """Forecast with Holt-Winters exponential smoothing.

    Holt-Winters smoothing iteratively updates level, trend, and seasonal
    components using exponentially decaying weights.  The additive trend and
    seasonality configuration works well for demand series where seasonal
    effects are roughly constant in magnitude.

    Parameters
    ----------
    train_series:
        In-sample history used to fit level, trend, and seasonality components.
    test_series:
        Future horizon for which predictions are requested.
    seasonal_periods:
        Number of observations per season (7 for weekly seasonality on daily data).

    Returns
    -------
    pandas.Series
        Holt-Winters forecasts aligned to the ``test_series`` index.
    """

    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
    )
    fitted = model.fit(optimized=True)
    forecast = fitted.forecast(len(test_series))
    return pd.Series(forecast, index=test_series.index, name="holt_winters")


def arima_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    order: Tuple[int, int, int] = (2, 1, 2),
) -> pd.Series:
    """Forecast with an ARIMA model of specified order.

    Autoregressive Integrated Moving Average models combine differencing to
    remove trends with autoregressive and moving average terms that explain
    autocorrelation in the differenced series.  The provided order corresponds
    to AR(2), I(1), MA(2), a common starting point for moderately complex
    demand patterns.

    Parameters
    ----------
    train_series:
        Differenced model will be fit to this historical data.
    test_series:
        Out-of-sample window to forecast.
    order:
        Tuple of ``(p, d, q)`` specifying autoregressive order, differencing degree,
        and moving-average order.

    Returns
    -------
    pandas.Series
        ARIMA predictions indexed by the test period dates.
    """

    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test_series))
    return pd.Series(forecast, index=test_series.index, name="arima")


def sarima_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
) -> pd.Series:
    """Forecast with a seasonal ARIMA (SARIMA) model.

    SARIMA expands ARIMA by including seasonal autoregressive and moving
    average components that repeat every ``seasonal_order[-1]`` periods.  This
    makes the model well-suited for demand exhibiting weekly cyclicality such
    as sales volumes influenced by day-of-week patterns.

    Parameters
    ----------
    train_series:
        Historical values used to estimate seasonal autoregressive behavior.
    test_series:
        Time range that requires forecasts.
    order:
        Non-seasonal ``(p, d, q)`` configuration for the SARIMA model.
    seasonal_order:
        Seasonal ``(P, D, Q, s)`` specification capturing periodic structure with
        period ``s``.

    Returns
    -------
    pandas.Series
        SARIMA predictions aligned to the ``test_series`` index.
    """

    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test_series))
    return pd.Series(forecast, index=test_series.index, name="sarima")


def prophet_forecast(train_series: pd.Series, test_series: pd.Series) -> pd.Series:
    """Forecast with Facebook's Prophet additive time series model.

    Prophet decomposes the signal into trend, seasonality, and holiday effects
    using piecewise linear or logistic growth curves with automatic changepoint
    detection.  It is robust to missing data and handles seasonality via a
    Fourier series representation, making it convenient for business time
    series.

    Parameters
    ----------
    train_series:
        Historical demand indexed by date for model fitting.
    test_series:
        Future window for which predictions will be produced.

    Returns
    -------
    pandas.Series
        Prophet forecasts indexed by the same dates as ``test_series``.
    """

    train_df = train_series.reset_index()
    train_df.columns = ["ds", "y"]

    model = Prophet()
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_series), freq="D")
    forecast = model.predict(future)
    predicted = forecast.tail(len(test_series))["yhat"].to_numpy()
    return pd.Series(predicted, index=test_series.index, name="prophet")


def _build_time_features(series: pd.Series) -> pd.DataFrame:
    """Create calendar, lag, and rolling statistics for machine learning models.

    Parameters
    ----------
    series:
        Full continuous series (train + test) used to engineer features.

    Returns
    -------
    pandas.DataFrame
        Feature matrix with calendar attributes, lag values, and rolling stats.
    """

    frame = series.sort_index().reset_index()
    frame.columns = ["date", "y"]

    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_month"] = frame["date"].dt.day
    frame["month"] = frame["date"].dt.month
    frame["quarter"] = frame["date"].dt.quarter

    for lag in (1, 7, 14, 30):
        frame[f"lag_{lag}"] = frame["y"].shift(lag)

    for window in (7, 14, 30):
        frame[f"rolling_mean_{window}"] = frame["y"].rolling(window).mean()
        frame[f"rolling_std_{window}"] = frame["y"].rolling(window).std()

    return frame.dropna().reset_index(drop=True)


def _train_tree_based_model(
    model,
    train_series: pd.Series,
    test_series: pd.Series,
) -> pd.Series:
    """Shared helper that fits a regression model on engineered features.

    Parameters
    ----------
    model:
        Any scikit-learn compatible regressor implementing ``fit`` and ``predict``.
    train_series:
        In-sample data used to compute features and train the model.
    test_series:
        Out-of-sample horizon to forecast.

    Returns
    -------
    pandas.Series
        Predictions produced by the trained regressor, indexed by test dates.
    """

    full_series = pd.concat([train_series, test_series])
    feature_frame = _build_time_features(full_series)

    train_end_date = train_series.index[-1]
    train_mask = feature_frame["date"] <= train_end_date
    test_mask = feature_frame["date"].isin(test_series.index)

    train_df = feature_frame.loc[train_mask]
    test_df = feature_frame.loc[test_mask].sort_values("date")

    X_train = train_df.drop(columns=["date", "y"])
    y_train = train_df["y"]
    X_test = test_df.drop(columns=["date", "y"])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predicted_series = pd.Series(predictions, index=test_df["date"], name=model.__class__.__name__.lower())
    return predicted_series.reindex(test_series.index)


def random_forest_forecast(train_series: pd.Series, test_series: pd.Series) -> pd.Series:
    """Forecast with a Random Forest regressor using engineered time features.

    The Random Forest ensemble aggregates predictions from many decision trees
    trained on bootstrapped samples.  By averaging de-correlated trees, the
    model captures non-linear interactions between calendar effects, lagged
    demand, and rolling statistics while remaining resilient to overfitting.

    Parameters
    ----------
    train_series:
        Historical portion of the time series for training the ensemble.
    test_series:
        Future horizon where predictions are required.

    Returns
    -------
    pandas.Series
        Random Forest predictions indexed by the ``test_series`` dates.
    """

    model = RandomForestRegressor(n_estimators=500, random_state=42)
    return _train_tree_based_model(model, train_series, test_series)


def gradient_boosting_forecast(train_series: pd.Series, test_series: pd.Series) -> pd.Series:
    """Forecast with Gradient Boosting regression over time-based predictors.

    Gradient Boosting fits an ensemble of shallow decision trees sequentially,
    each correcting the residual errors of the previous trees.  This boosting
    strategy often excels on structured tabular data where subtle interactions
    between lag and calendar features drive the demand signal.

    Parameters
    ----------
    train_series:
        In-sample segment used to engineer features and fit the gradient boosting model.
    test_series:
        Future segment for which forecasts are generated.

    Returns
    -------
    pandas.Series
        Gradient Boosting predictions aligned with the ``test_series`` index.
    """

    model = GradientBoostingRegressor(random_state=42)
    return _train_tree_based_model(model, train_series, test_series)


__all__ = [
    "prepare_daily_sales",
    "train_test_split",
    "moving_average_forecast",
    "exponential_smoothing_forecast",
    "arima_forecast",
    "sarima_forecast",
    "prophet_forecast",
    "random_forest_forecast",
    "gradient_boosting_forecast",
]

