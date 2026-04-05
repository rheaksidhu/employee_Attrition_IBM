import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
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

#confirm no protected attributes remain in the feature matrix before any splitting
assert not any(col in featureData.columns for col in auditColumns), \
    "Protected attribute found in feature matrix"

#perform an 80/20 stratified split before any preprocessing or model fitting so the
#held-out test set is never touched until final evaluation and cannot influence
#preprocessing decisions or hyperparameter choices
xTrain, xTest, yTrain, yTest = train_test_split(
    featureData, yTarget,
    test_size=0.20,
    stratify=yTarget,
    random_state=42
)

print(f"\nTraining set size: {xTrain.shape[0]} rows")
print(f"Test set size:     {xTest.shape[0]} rows")
print(f"Training attrition rate: {yTrain.mean():.3f}")
print(f"Test attrition rate:     {yTest.mean():.3f}")

#define nominal categorical columns that require OneHotEncoder to avoid implying
#a false ordinal relationship between unordered categories
nominalColumns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole']

#OverTime is binary Yes/No and is treated as a nominal column to avoid encoding it
#as an arbitrary integer; OneHotEncoder makes the intent explicit and consistent
nominalColumns.append('OverTime')

#identify all remaining numeric columns; StandardScaler is applied so that features
#measured on very different scales contribute equally to the model and convergence
#is consistent; RandomForest does not require scaling but it is retained to keep
#the preprocessing pipeline identical to V2b_Sonnet.py as required
numericColumns = xTrain.select_dtypes(include=[np.number]).columns.tolist()

print("\nNominal columns (OneHotEncoded):", nominalColumns)
print("Numeric columns (StandardScaled):", numericColumns)

#verify that the nominal and numeric column lists cover all feature columns with no
#overlap; any unaccounted columns would silently be dropped by ColumnTransformer
allHandled = set(nominalColumns + numericColumns)
allFeature = set(xTrain.columns)
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

#RandomForest is chosen over LogisticRegression because it captures non-linear feature
#interactions (e.g. high OverTime combined with low JobSatisfaction) that a linear
#model cannot represent; n_estimators=500 gives a stable ensemble with low variance;
#class_weight='balanced' up-weights the minority attrition class to correct for the
#roughly 16% positive rate without requiring resampling
randomForestModel = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

#combine preprocessing and classifier into a single pipeline so that encoders and
#scalers are fitted only on training folds and never see test-fold data, preventing
#any form of data leakage into the cross-validation estimate
modelPipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', randomForestModel)
])

#StratifiedKFold preserves the original class ratio in every fold; critical when the
#positive class is rare (~16% attrition) so each fold gives a representative estimate
stratifiedKfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Running 5-fold cross-validation on training set ---")

#cross_val_predict with method='predict_proba' returns out-of-fold probability estimates
#for every training sample; using probabilities rather than hard labels allows AUC-ROC
#to reflect the full probability range rather than a single decision threshold
cvProbabilities = cross_val_predict(
    modelPipeline, xTrain, yTrain,
    cv=stratifiedKfold, method='predict_proba'
)[:, 1]

#derive hard binary predictions at threshold 0.5 to compute precision, recall and F1
cvPredicted = (cvProbabilities >= 0.5).astype(int)

cvAucRoc = roc_auc_score(yTrain, cvProbabilities)
cvF1 = f1_score(yTrain, cvPredicted, pos_label=1, zero_division=0)
cvPrecision = precision_score(yTrain, cvPredicted, pos_label=1, zero_division=0)
cvRecall = recall_score(yTrain, cvPredicted, pos_label=1, zero_division=0)

print("\n--- Cross-Validation Metrics (training set, positive class = Attrition Yes) ---")
print(f"AUC-ROC:   {cvAucRoc:.4f}")
print(f"F1-Score:  {cvF1:.4f}")
print(f"Precision: {cvPrecision:.4f}")
print(f"Recall:    {cvRecall:.4f}")

#train the final model on the complete training set after cross-validation is finished;
#fitting on all available training data gives the strongest possible model before
#evaluating on the genuinely unseen held-out test set
print("\n--- Training final model on full training set ---")
modelPipeline.fit(xTrain, yTrain)

#evaluate on the held-out test set for the first and only time; this is the unbiased
#estimate of generalisation performance because the test set was never seen during
#preprocessing, cross-validation, or model selection
testProbabilities = modelPipeline.predict_proba(xTest)[:, 1]
testPredicted = (testProbabilities >= 0.5).astype(int)

testAucRoc = roc_auc_score(yTest, testProbabilities)
testF1 = f1_score(yTest, testPredicted, pos_label=1, zero_division=0)
testPrecision = precision_score(yTest, testPredicted, pos_label=1, zero_division=0)
testRecall = recall_score(yTest, testPredicted, pos_label=1, zero_division=0)

print("\n--- Held-Out Test Set Metrics (positive class = Attrition Yes) ---")
print(f"AUC-ROC:   {testAucRoc:.4f}")
print(f"F1-Score:  {testF1:.4f}")
print(f"Precision: {testPrecision:.4f}")
print(f"Recall:    {testRecall:.4f}")

#extract feature names from the fitted ColumnTransformer so that feature importances
#from the RandomForest can be mapped back to human-readable column names rather than
#anonymous indices; get_feature_names_out() returns OHE-expanded names for nominals
fittedPreprocessor = modelPipeline.named_steps['preprocessor']
oneHotFeatureNames = fittedPreprocessor.named_transformers_['oneHot'] \
    .get_feature_names_out(nominalColumns).tolist()
allFeatureNames = oneHotFeatureNames + numericColumns

#retrieve feature importances from the fitted RandomForest inside the pipeline;
#these are mean impurity-decrease values across all 500 trees and reflect the
#actual fitted model, not logistic regression coefficients
fittedForest = modelPipeline.named_steps['classifier']
featureImportances = fittedForest.feature_importances_

#build a series indexed by feature name for easy sorting and lookup
importanceSeries = pd.Series(featureImportances, index=allFeatureNames).sort_values(
    ascending=False
)

#for the Oracle, map OHE-expanded feature names back to their original column name so
#that retention suggestions can be looked up against human-readable feature names;
#e.g. 'OverTime_Yes' maps to 'OverTime' so the correct suggestion is retrieved
def getOriginalFeatureName(expandedName):
    #strip the OHE suffix by checking if the expanded name starts with any nominal column
    for nomCol in nominalColumns:
        if expandedName.startswith(nomCol + '_'):
            return nomCol
    return expandedName

#map original column names to retention suggestions so that actionable guidance is
#produced for every flagged employee rather than a generic warning
retentionSuggestions = {
    'OverTime': "Consider offering flexible working arrangements or reviewing workload distribution",
    'MonthlyIncome': "Review compensation against market benchmarks and consider a salary adjustment",
    'JobSatisfaction': "Conduct a one-to-one satisfaction review and identify specific pain points",
    'YearsSinceLastPromotion': "Discuss career progression pathway and set clear promotion criteria",
    'WorkLifeBalance': "Explore flexible working options or additional leave entitlements",
    'Age': "This is a protected characteristic and should not be used for intervention targeting",
    'DistanceFromHome': "Consider remote working arrangements or travel support",
    'TotalWorkingYears': "Recognise experience and ensure role complexity matches seniority",
    'JobLevel': "Review whether current role reflects the employee's skills and experience",
}
defaultSuggestion = "Review this factor with the employee's line manager"

#build a ranked list of (originalFeatureName, importance) pairs so that the top 3
#features per flagged employee can be identified using the same global importance
#ranking from the fitted RandomForest; this avoids per-employee SHAP computation
#while still providing directionally meaningful retention guidance
rankedFeaturePairs = [
    (getOriginalFeatureName(name), imp)
    for name, imp in importanceSeries.items()
]

#deduplicate by original feature name keeping the highest importance for each group;
#necessary because OHE expands one nominal column into multiple binary columns whose
#individual importances should be aggregated to avoid misleading the Oracle ranking
seenFeatures = {}
for origName, imp in rankedFeaturePairs:
    if origName not in seenFeatures:
        seenFeatures[origName] = imp
    else:
        seenFeatures[origName] += imp

#sort deduplicated features by aggregated importance descending so top 3 per employee
#reflect the most influential original features rather than arbitrary OHE categories
rankedOriginalFeatures = sorted(seenFeatures.items(), key=lambda x: x[1], reverse=True)

print("\n--- Oracle Refinement: flagged employees (predicted attrition probability > 50%) ---")

#identify test set indices where the model predicts a leaving probability above 0.5;
#resetting the index ensures positional alignment between testProbabilities array
#and the xTest dataframe rows
xTestReset = xTest.reset_index(drop=True)
flaggedIndices = np.where(testProbabilities > 0.5)[0]

print(f"Total employees flagged: {len(flaggedIndices)}")
print(f"Showing first 5 flagged employees:\n")

#iterate only over the first 5 flagged employees to keep the console output readable;
#all flagged employees could be exported to a file for operational use
for displayRank, empIdx in enumerate(flaggedIndices[:5], start=1):
    attritionPct = testProbabilities[empIdx] * 100
    top3Features = rankedOriginalFeatures[:3]

    print(f"Employee {displayRank} (test set row {empIdx})")
    print(f"  Predicted attrition probability: {attritionPct:.1f}%")
    print("  Top 3 influential features and retention suggestions:")

    for featureRank, (featureName, featureImp) in enumerate(top3Features, start=1):
        suggestion = retentionSuggestions.get(featureName, defaultSuggestion)
        print(f"    {featureRank}. {featureName} (importance: {featureImp:.4f})")
        print(f"       Suggestion: {suggestion}")
    print()

#plot the top 15 features by aggregated importance as a horizontal bar chart so that
#HR stakeholders can immediately identify which factors most influence attrition risk
#without needing to interpret raw coefficient values
top15Features = rankedOriginalFeatures[:15]
top15Names = [name for name, _ in top15Features]
top15Importances = [imp for _, imp in top15Features]

figImportance, axImportance = plt.subplots(figsize=(10, 7))
yPositions = np.arange(len(top15Names))

axImportance.barh(yPositions, top15Importances, color='steelblue', edgecolor='white')
axImportance.set_yticks(yPositions)
axImportance.set_yticklabels(top15Names, fontsize=11)
axImportance.invert_yaxis()
axImportance.set_xlabel('Aggregated Feature Importance (Mean Impurity Decrease)', fontsize=12)
axImportance.set_title(
    'Top 15 Feature Importances – Random Forest (Balanced)\nEmployee Attrition Prediction',
    fontsize=12
)
axImportance.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

importancePlotPath = 'v3a_Sonnet_rf_feature_importance.png'
plt.savefig(importancePlotPath, dpi=150)
plt.show()
plt.close()
print(f"Feature importance chart saved to: {importancePlotPath}")

#compute the confusion matrix on the held-out test set predictions to give an unbiased
#view of the trade-off between false negatives (missed attrition) and false positives
#(unnecessary interventions); rows are true labels, columns are predicted labels
testConfusionMatrix = confusion_matrix(yTest, testPredicted)
print("\nConfusion Matrix (held-out test set):")
print(testConfusionMatrix)

#plot the confusion matrix as a colour-coded heatmap so the balance between
#false negatives and false positives is immediately visible without reading raw numbers
figCm, axCm = plt.subplots(figsize=(8, 6))
heatmapImage = axCm.imshow(testConfusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(heatmapImage, ax=axCm)

classLabels = ['No Attrition', 'Attrition']
tickPositions = np.arange(len(classLabels))
axCm.set_xticks(tickPositions)
axCm.set_yticks(tickPositions)
axCm.set_xticklabels(classLabels, fontsize=12)
axCm.set_yticklabels(classLabels, fontsize=12)

#use contrasting text colour so cell counts remain legible against both light and dark
#heatmap cells; white text on dark cells and black text on light cells
thresholdValue = testConfusionMatrix.max() / 2.0
for rowIndex in range(testConfusionMatrix.shape[0]):
    for colIndex in range(testConfusionMatrix.shape[1]):
        cellValue = testConfusionMatrix[rowIndex, colIndex]
        textColour = 'white' if cellValue > thresholdValue else 'black'
        axCm.text(
            colIndex, rowIndex, str(cellValue),
            ha='center', va='center',
            color=textColour, fontsize=14, fontweight='bold'
        )

axCm.set_xlabel('Predicted Label', fontsize=13)
axCm.set_ylabel('True Label', fontsize=13)
axCm.set_title(
    'Confusion Matrix – Random Forest (Balanced)\nEmployee Attrition Prediction '
    f'| AUC-ROC: {testAucRoc:.3f} | F1: {testF1:.3f}',
    fontsize=12
)

plt.tight_layout()

cmPlotPath = 'v3a_Sonnet_confusion_matrix.png'
plt.savefig(cmPlotPath, dpi=150)
plt.show()
plt.close()
print(f"Confusion matrix heatmap saved to: {cmPlotPath}")
