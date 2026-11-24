# Results: heart-failure mortality

*Generated automatically by `src/run.py`. Runtime 4.4 min.*

Data: 299 patients, 96 deaths. Validation: 20 × repeated stratified 5-fold CV. Brackets are the 2.5–97.5 percentile range of the per-repeat mean (split variability).

| Model | ROC AUC | PR AUC | Brier | Log loss | Calibration slope | Calibration intercept |
|---|---|---|---|---|---|---|
| Logistic regression | 0.772 [0.753, 0.791] | 0.640 [0.610, 0.669] | 0.175 [0.168, 0.182] | 0.533 [0.513, 0.551] | 0.890 [0.774, 1.020] | 0.009 [-0.024, 0.039] |
| Gradient boosting | 0.759 [0.738, 0.773] | 0.627 [0.603, 0.656] | 0.182 [0.175, 0.190] | 0.570 [0.552, 0.593] | 0.617 [0.572, 0.695] | 0.073 [-0.007, 0.153] |
| Random forest | 0.793 [0.778, 0.804] | 0.668 [0.649, 0.689] | 0.167 [0.164, 0.171] | 0.508 [0.501, 0.515] | 1.331 [1.232, 1.518] | 0.010 [-0.024, 0.034] |
| Small neural net (MLP) | 0.773 [0.761, 0.785] | 0.639 [0.615, 0.658] | 0.187 [0.181, 0.190] | 0.557 [0.543, 0.563] | 2.581 [2.106, 3.103] | 0.008 [-0.014, 0.046] |

Prevalence of death is 0.321, so a PR AUC near that value means no skill.
A calibration slope below 1 means predictions are too extreme (overfitting); above 1, too timid.

## Leakage check

Logistic regression mean ROC AUC without `time`: **0.775**; with `time`: **0.879**. The jump comes from follow-up length, which is known only after the outcome, so models using it are not usable at the point of care.

## Most important features (mean |SHAP|, gradient boosting)

| Feature | mean abs SHAP |
|---|---|
| serum_creatinine | 0.633 |
| ejection_fraction | 0.603 |
| age | 0.409 |
| creatinine_phosphokinase | 0.380 |
| serum_sodium | 0.198 |
| platelets | 0.193 |
| high_blood_pressure | 0.123 |
| sex | 0.098 |
| diabetes | 0.055 |
| smoking | 0.022 |
| anaemia | 0.000 |
