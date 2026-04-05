# Employee Attrition Prediction & Fairness Audit

This project trains a supervised binary classification model to predict employee attrition using the IBM HR Analytics dataset, and audits whether predictions correlate with protected characteristics under the UK Equality Act 2010.

A True Positive (TP) here means the model predicted an employee would leave and they actually did

---

## Dataset
IBM HR Analytics Employee Attrition & Performance: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)

---

## Prompt Engineering
The meta prompt development process (prompts 1a–1f) is documented in [Meta_Prompt_Development.md](Meta_Prompt_Development.md)

---

## Version Branches
| Branch | Description |
|--------|-------------|
| [v1_Baseline](../../tree/v1_Baseline) | Initial AI-generated logistic regression baseline (ChatGPT & Claude) |
| [v2_Logistic_Regression](../../tree/v2_Logistic_Regression) | Improved logistic regression with proper preprocessing and cross-validation |
| [v3_Random_Forest](../../tree/v3_Random_Forest) | Random Forest with SHAP explanations and threshold tuning |
| [v4_XGBoost](../../tree/v4_XGBoost) | Final XGBoost model with correlation analysis and fairness audit |

---

## Initial Code
The two unmodified AI-generated baseline scripts are:
- [attritionV1a_ChatGPT_initial.py](../../blob/v1_Baseline/attritionV1a_ChatGPT_initial.py): unmodified GPT-5 output
- [attritionV1b_Claude_initial.py](../../blob/v1_Baseline/attritionV1b_Claude_initial.py): unmodified Claude Sonnet 4.6 output

## Final Code
The final model is:
- [attritionV4a_Sonnet_initial.py](../../blob/v4_XGBoost/attritionV4a_Sonnet_initial.py): XGBoost with SHAP explanations and fairness audit

---

## Module
BSC Applied Artificial Intelligence


SPC4004 Exploring AI: Understanding and Applications 

Queen Mary University of London
