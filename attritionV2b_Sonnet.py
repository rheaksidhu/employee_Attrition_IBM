import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)

#load raw data before any transformation so column names can be verified
rawData = pd.read_csv('IBM_HR_Analytics_Employee_Attrition_and_Performance.csv')

print("Dataset shape:", rawData.shape)
print("\nColumn names:")
print(rawData.columns.tolist())

#drop columns that carry no predictive signal: EmployeeCount and StandardHours are
#constants, EmployeeNumber is an arbitrary ID, Over18 has a single value across all rows
constantColumns = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
rawData = rawData.drop(columns=constantColumns)

#isolate protected demographic attributes into a separate audit dataframe before
#building the feature matrix so they cannot leak into the model and cause indirect
#discrimination; these columns are preserved for fairness audits only
auditColumns = ['Age', 'Gender', 'MaritalStatus']
auditData = rawData[auditColumns].copy()
print("\nAudit dataframe shape (protected attributes):", auditData.shape)
print("Audit columns:", auditData.columns.tolist())

#encode the binary target as 1 (Yes = attrition) and 0 (No = retained) before splitting
#so that positive-class metrics unambiguously refer to employees who left
yTarget = (rawData['Attrition'] == 'Yes').astype(int)
print("\nAttrition class distribution:")
print(yTarget.value_counts())

#build the feature matrix excluding the target and all audit columns; this guarantees
#that Age, Gender and MaritalStatus are absent from training
excludeColumns = ['Attrition'] + auditColumns
featureData = rawData.drop(columns=excludeColumns)

#confirm no protected attributes remain in the feature matrix
assert not any(col in featureData.columns for col in auditColumns), \
    "Protected attribute found in feature matrix"

#define nominal categorical columns that require OneHotEncoder to avoid implying
#a false ordinal relationship between unordered categories
nominalColumns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole']

#OverTime is binary Yes/No and is treated as a nominal column to avoid encoding it
#as an arbitrary integer (LabelEncoder would assign 0/1 which happens to be valid here
#but OneHotEncoder makes the intent explicit and consistent with the other nominals)
nominalColumns.append('OverTime')

#identify all remaining numeric columns; StandardScaler is applied so that features
#measured on very different scales (e.g. MonthlyIncome vs JobLevel) contribute equally
#to the logistic regression loss function and convergence is faster
numericColumns = featureData.select_dtypes(include=[np.number]).columns.tolist()

print("\nNominal columns (OneHotEncoded):", nominalColumns)
print("Numeric columns (StandardScaled):", numericColumns)

#verify that the nominal and numeric column lists cover all feature columns with no
#overlap; any unaccounted columns would silently be dropped by ColumnTransformer
allHandled = set(nominalColumns + numericColumns)
allFeature = set(featureData.columns)
unhandled = allFeature - allHandled
print("\nUnhandled columns (should be empty):", unhandled)
assert len(unhandled) == 0, f"Columns not assigned to any transformer: {unhandled}"

#build a ColumnTransformer that applies OneHotEncoder and StandardScaler in one step;
#handle_unknown='ignore' prevents errors if a category unseen during training appears
#during cross-validation folds; remainder='drop' discards any stray columns safely
preprocessor = ColumnTransformer(
    transformers=[
        ('oneHot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominalColumns),
        ('scaler', StandardScaler(), numericColumns)
    ],
    remainder='drop'
)

#class_weight='balanced' instructs the solver to up-weight the minority class (Attrition=Yes)
#inversely proportional to its frequency; this prevents the model from simply predicting
#No for every sample to achieve high accuracy on an imbalanced dataset
logisticModel = LogisticRegression(
    class_weight='balanced',
    max_iter=5000,
    solver='saga',
    random_state=42
)

#combine preprocessing and model into a single pipeline so that the scaler and encoder
#are fitted only on training folds and never see test-fold data, preventing data leakage
modelPipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', logisticModel)
])

#StratifiedKFold preserves the original class ratio in every fold; this is critical
#when one class is rare (roughly 16% attrition here) so each fold is representative
stratifiedKfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#cross_val_predict with method='predict_proba' returns out-of-fold probability estimates
#for every sample; using probabilities rather than hard labels allows AUC-ROC to be
#computed on the full probability range rather than a single threshold
yProbabilities = cross_val_predict(
    modelPipeline, featureData, yTarget,
    cv=stratifiedKfold, method='predict_proba'
)[:, 1]

#derive hard binary predictions at threshold 0.5 for precision, recall and F1
yPredicted = (yProbabilities >= 0.5).astype(int)

#evaluate exclusively on the positive class (Attrition=Yes, label=1); accuracy is
#omitted because it is misleading under class imbalance and does not capture how well
#the model identifies the minority class that is operationally most important
aucRoc = roc_auc_score(yTarget, yProbabilities)
f1 = f1_score(yTarget, yPredicted, pos_label=1, zero_division=0)
precision = precision_score(yTarget, yPredicted, pos_label=1, zero_division=0)
recall = recall_score(yTarget, yPredicted, pos_label=1, zero_division=0)

print("\n--- Evaluation Metrics (positive class = Attrition Yes) ---")
print(f"AUC-ROC:   {aucRoc:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

#compute the confusion matrix using out-of-fold predictions to give an unbiased
#estimate of generalisation performance; rows are true labels, columns are predicted
confusionMatrix = confusion_matrix(yTarget, yPredicted)
print("\nConfusion Matrix:")
print(confusionMatrix)

#plot the confusion matrix as a colour-coded heatmap so that the balance between
#false negatives (missed attrition) and false positives (unnecessary intervention)
#is immediately visible without reading raw numbers
figurePlot, axisPlot = plt.subplots(figsize=(8, 6))

heatmapImage = axisPlot.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(heatmapImage, ax=axisPlot)

classLabels = ['No Attrition', 'Attrition']
tickPositions = np.arange(len(classLabels))
axisPlot.set_xticks(tickPositions)
axisPlot.set_yticks(tickPositions)
axisPlot.set_xticklabels(classLabels, fontsize=12)
axisPlot.set_yticklabels(classLabels, fontsize=12)

#use contrasting text colour so cell counts remain legible against both light and dark
#heatmap cells; white text on dark cells and black text on light cells
thresholdValue = confusionMatrix.max() / 2.0
for rowIndex in range(confusionMatrix.shape[0]):
    for colIndex in range(confusionMatrix.shape[1]):
        cellValue = confusionMatrix[rowIndex, colIndex]
        textColour = 'white' if cellValue > thresholdValue else 'black'
        axisPlot.text(
            colIndex, rowIndex, str(cellValue),
            ha='center', va='center',
            color=textColour, fontsize=14, fontweight='bold'
        )

axisPlot.set_xlabel('Predicted Label', fontsize=13)
axisPlot.set_ylabel('True Label', fontsize=13)
axisPlot.set_title(
    'Confusion Matrix – Logistic Regression (Balanced)\nEmployee Attrition Prediction '
    f'| AUC-ROC: {aucRoc:.3f} | F1: {f1:.3f}',
    fontsize=12
)

plt.tight_layout()

#save the heatmap as a PNG so it can be embedded in the assessment report without
#requiring the reader to re-run the script
outputFilePath = 'v2b_Sonnet_confusion_matrix.png'
plt.savefig(outputFilePath, dpi=150)
plt.show()
plt.close()

print(f"\nConfusion matrix heatmap saved to: {outputFilePath}")
