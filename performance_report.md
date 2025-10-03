# Sales Forecasting Model Performance Report

> This report is generated automatically by `model_evaluation.evaluate_models`. Run the evaluation workflow after producing predictions to refresh the metrics, rankings, and insights below.

## Model Rankings
- Pending evaluation – execute the automated evaluation pipeline to compute MAE, RMSE, MAPE, and R² for each forecasting model.

## Metric Explanations
- **MAE (Mean Absolute Error):** Average magnitude of the errors without considering their direction.
- **RMSE (Root Mean Squared Error):** Quadratic mean of residuals that penalizes larger errors more heavily.
- **MAPE (Mean Absolute Percentage Error):** Average absolute percentage difference between predicted and actual values.
- **R² Score:** Proportion of variance in the target explained by the model (closer to 1 indicates better fit).

## Model Recommendations
- The evaluation pipeline will identify the lowest-RMSE model and flag it as the primary recommendation once metrics are available.
- Monitor MAE and RMSE weekly; retrain the models if error levels degrade materially.
- Consider blending the top two models if their error profiles are complementary once evaluation results are generated.

## Sales Data Insights
- Descriptive insights (total quantity, average daily demand, peak days, and recent trend) are added when the evaluation workflow is executed with actual sales data.
