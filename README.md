# Employee Attrition Prediction & Fairness Audit

This project trains a supervised binary classification model to predict employee attrition using the IBM HR Analytics dataset, and audits whether predictions correlate with protected characteristics under the UK Equality Act 2010.

## Dataset
IBM HR Analytics Employee Attrition & Performance — [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)  
1,470 employee records, 35 features covering behavioural and demographic attributes.

## Prompt Engineering
The meta prompt development process (prompts 1a–1f) is documented in [META_PROMPT_DEVELOPMENT.md](META_PROMPT_DEVELOPMENT.md)

## Version Branches
| Branch | Description |
|--------|-------------|
| [v1-baseline](../../tree/v1-baseline) | Initial AI-generated logistic regression baseline (ChatGPT & Claude) |
| [v2-logistic-regression](../../tree/v2-logistic-regression) | Improved logistic regression with proper preprocessing and cross-validation |
| [v3-random-forest](../../tree/v3-random-forest) | Random Forest with SHAP explanations and threshold tuning |
| [v4-xgboost-final](../../tree/v4-xgboost-final) | Final XGBoost model with fairness audit |

## Module
SPC4004 Exploring AI: Understanding and Applications — Queen Mary University of London
