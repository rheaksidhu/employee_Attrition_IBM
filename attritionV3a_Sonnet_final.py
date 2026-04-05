import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)

#load raw data before any transformation so column names can be verified
rawData = pd.read_csv('IBM_HR_Analytics_Employee_Attrition_and_Performance.csv')

print("Dataset shape:", rawData.shape)
print("\nColumn names:")
print(rawData.columns.tolist())

#drop columns that carry no predictive signal: EmployeeCount and StandardHours are constants, EmployeeNumber is an arbitrary ID, Over18 has a single value across all rows
constantColumns = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
rawData = rawData.drop(columns=constantColumns)

#isolate protected demographic attributes into a separate audit dataframe before building the feature matrix so they cannot leak into the model and cause indirect discrimination; these columns are preserved for fairness audits only
auditColumns = ['Age', 'Gender', 'MaritalStatus']
auditData = rawData[auditColumns].copy()
print("\nAudit dataframe shape (protected attributes):", auditData.shape)
print("Audit columns:", auditData.columns.tolist())

#encode the binary target as 1 (Yes = attrition) and 0 (No = retained) before splitting so that positive-class metrics unambiguously refer to employees who left
yTarget = (rawData['Attrition'] == 'Yes').astype(int)
print("\nAttrition class distribution:")
print(yTarget.value_counts())

#build the feature matrix excluding the target and all audit columns; this guarantees that Age, Gender and MaritalStatus are absent from training
excludeColumns = ['Attrition'] + auditColumns
featureData = rawData.drop(columns=excludeColumns)

#confirm no protected attributes remain in the feature matrix before any splitting
assert not any(col in featureData.columns for col in auditColumns), \
    "Protected attribute found in feature matrix"

#perform an 85/15 stratified split before any preprocessing or model fitting so the held-out test set is never touched until final evaluation and cannot influence preprocessing decisions or hyperparameter choices; 85/15 split gives the model more training data which helps it learn the minority attrition class better
xTrain, xTest, yTrain, yTest = train_test_split(
    featureData, yTarget,
    test_size=0.15,
    stratify=yTarget,
    random_state=42
)

print(f"\nTraining set size: {xTrain.shape[0]} rows")
print(f"Test set size:     {xTest.shape[0]} rows")
print(f"Training attrition rate: {yTrain.mean():.3f}")
print(f"Test attrition rate:     {yTest.mean():.3f}")

#define nominal categorical columns that require OneHotEncoder to avoid implying a false ordinal relationship between unordered categories
nominalColumns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole']

#OverTime is binary Yes/No and is treated as a nominal column to avoid encoding it as an arbitrary integer; OneHotEncoder makes the intent explicit and consistent
nominalColumns.append('OverTime')

#identify all remaining numeric columns; StandardScaler is applied so that features measured on very different scales contribute equally and the preprocessing pipeline remains identical to prior versions as required
numericColumns = xTrain.select_dtypes(include=[np.number]).columns.tolist()

print("\nNominal columns (OneHotEncoded):", nominalColumns)
print("Numeric columns (StandardScaled):", numericColumns)

#verify that the nominal and numeric column lists cover all feature columns with no overlap; any unaccounted columns would silently be dropped by ColumnTransformer
allHandled = set(nominalColumns + numericColumns)
allFeature = set(xTrain.columns)
unhandled = allFeature - allHandled
print("\nUnhandled columns (should be empty):", unhandled)
assert len(unhandled) == 0, f"Columns not assigned to any transformer: {unhandled}"

#build a ColumnTransformer that applies OneHotEncoder and StandardScaler in one step; handle_unknown='ignore' prevents errors if a category unseen during training appears during cross-validation folds; remainder='drop' discards any stray columns safely
preprocessor = ColumnTransformer(
    transformers=[
        ('oneHot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominalColumns),
        ('scaler', StandardScaler(), numericColumns)
    ],
    remainder='drop'
)

#fit the preprocessor on training data only so that encoding and scaling statistics are derived solely from xTrain and never leak information from xTest
preprocessor.fit(xTrain)

#transform both splits using statistics fitted on xTrain only; this mirrors what a Pipeline would do but allows us to pass a plain numpy array to RandomizedSearchCV with a bare RandomForest rather than a pipeline, which is required so that CalibratedClassifierCV can wrap the forest without nesting Pipeline inside Pipeline
xTrainTransformed = preprocessor.transform(xTrain)
xTestTransformed = preprocessor.transform(xTest)

#recover feature names after OHE expansion so SHAP values can be mapped back to human-readable column names for the Oracle output and importance chart
oheFeatureNames = preprocessor.named_transformers_['oneHot'].get_feature_names_out(nominalColumns).tolist()
allFeatureNames = oheFeatureNames + numericColumns

#StratifiedKFold preserves the original class ratio in every fold; critical when the positive class is rare (~16% attrition) so each fold gives a representative estimate
stratifiedKfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#define the hyperparameter search space; recall is used as the scoring metric because missing an employee who will leave (false negative) is more costly than a false alarm; the parameter grid is optimised for RandomizedSearchCV sampling which covers broader search space efficiently
paramGrid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', 0.3],
    'max_samples': [0.8, None]
}

#base RandomForest with balanced class weight to correct for the ~16% positive rate; RandomizedSearchCV will replace these parameters with the best found during search
baseForest = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

#RandomizedSearchCV samples 40 random combinations from the parameter grid rather than exhaustively testing all combinations, finding near-optimal parameters in a fraction of the time while covering a broader search space; fitted only on xTrainTransformed so the test set is never passed here which guarantees the held-out evaluation remains completely unbiased
print("\n--- Running RandomizedSearchCV on training set only (this may take several minutes) ---")
randomSearch = RandomizedSearchCV(
    estimator=baseForest,
    param_distributions=paramGrid,
    n_iter=40,
    scoring='recall',
    cv=stratifiedKfold,
    n_jobs=-1,
    verbose=1,
    refit=True,
    random_state=42
)

randomSearch.fit(xTrainTransformed, yTrain)

print(f"\nBest parameters found: {randomSearch.best_params_}")
print(f"Best cross-validation recall score: {randomSearch.best_score_:.4f}")

#extract the best forest from RandomizedSearchCV for calibration; this is the estimator refit on the entire training set using the best parameters found during random search
bestForest = randomSearch.best_estimator_

#CalibratedClassifierCV wraps the best forest to produce probabilistic outputs that better match true likelihoods; sigmoid calibration fits a simpler S-curve and is more stable than isotonic when the minority class is small, which it is here at roughly 16% attrition
calibratedModel = CalibratedClassifierCV(
    estimator=bestForest,
    method='sigmoid',
    cv=stratifiedKfold
)

#fit calibration on training data; the calibrator learns to adjust probabilities so that for any predicted score, the observed frequency of positive class approaches that score
calibratedModel.fit(xTrainTransformed, yTrain)

#make predictions on the held-out test set for evaluation; test set has never been seen by RandomizedSearchCV, the forest, or the calibrator
testProbabilities = calibratedModel.predict_proba(xTestTransformed)[:, 1]

#apply a 0.3 threshold chosen to optimise recall; employees with predicted probability >= 0.3 are flagged for retention efforts
testPredicted03 = (testProbabilities >= 0.3).astype(int)

#compute evaluation metrics at the 0.3 threshold; recall is prioritised because missing an employee who will leave is more costly than alerting the business to a false positive
testAucRoc = roc_auc_score(yTest, testProbabilities)
testF1at03 = f1_score(yTest, testPredicted03)
testPrecisionAt03 = precision_score(yTest, testPredicted03)
testRecallAt03 = recall_score(yTest, testPredicted03)

print("\n--- Test Set Performance (held-out data, never touched during tuning) ---")
print(f"AUC-ROC (all thresholds): {testAucRoc:.4f}")
print(f"Precision at 0.3 threshold: {testPrecisionAt03:.4f}")
print(f"Recall at 0.3 threshold: {testRecallAt03:.4f}")
print(f"F1-Score at 0.3 threshold: {testF1at03:.4f}")

#SHAP explainer computes Shapley values which quantify the contribution of each feature to the model's prediction for each sample; this allows us to explain why a specific employee was flagged and which factors matter most globally
shapExplainer = shap.TreeExplainer(bestForest)
rawShapValues = shapExplainer.shap_values(xTestTransformed)

#handle both legacy and modern SHAP output formats because shap library versions may differ
if isinstance(rawShapValues, list):
    #legacy shap behaviour: list of [class0_array, class1_array]
    shapValues = rawShapValues[1]
elif rawShapValues.ndim == 3:
    #modern shap behaviour: single 3D array where axis 2 is the class dimension
    shapValues = rawShapValues[:, :, 1]
else:
    shapValues = rawShapValues

#wrap SHAP values in a DataFrame indexed by feature name for easy per-row lookup
shapDf = pd.DataFrame(shapValues, columns=allFeatureNames)

#map OHE-expanded feature names back to their original column name so that retention suggestions can be looked up against human-readable feature names; e.g. 'OverTime_Yes' maps to 'OverTime' so the correct suggestion is retrieved
def getOriginalFeatureName(expandedName):
    #strip the OHE suffix by checking if the expanded name starts with any nominal column
    for nomCol in nominalColumns:
        if expandedName.startswith(nomCol + '_'):
            return nomCol
    return expandedName

#map original column names to retention suggestions so actionable guidance is produced for every flagged employee rather than a generic warning
retentionSuggestions = {
    'OverTime': "Consider offering flexible working arrangements or reviewing workload distribution",
    'MonthlyIncome': "Review compensation against market benchmarks and consider a salary adjustment",
    'JobSatisfaction': "Conduct a one-to-one satisfaction review and identify specific pain points",
    'YearsSinceLastPromotion': "Discuss career progression pathway and set clear promotion criteria",
    'WorkLifeBalance': "Explore flexible working options or additional leave entitlements",
    'DistanceFromHome': "Consider remote working arrangements or travel support",
    'TotalWorkingYears': "Recognise experience and ensure role complexity matches seniority",
    'JobLevel': "Review whether current role reflects the employee skills and experience",
    'StockOptionLevel': "Review stock option allocation and long-term incentive package",
    'YearsAtCompany': "Recognise loyalty and ensure engagement and development opportunities reflect tenure",
}
defaultSuggestion = "Review this factor with the employee line manager"

#identify test set indices where predicted probability exceeds 0.3 threshold; resetting the index ensures positional alignment between the probability array and the SHAP dataframe rows
xTestReset = xTest.reset_index(drop=True)
flaggedIndices = np.where(testProbabilities >= 0.3)[0]

print(f"\n--- Oracle: employees flagged at threshold 0.3 ---")
print(f"Total employees flagged: {len(flaggedIndices)}")
print(f"Showing first 5 flagged employees:\n")

#iterate only over the first 5 flagged employees to keep the console output readable; all flagged employees could be exported to a file for operational use
for displayRank, empIdx in enumerate(flaggedIndices[:5], start=1):
    attritionPct = testProbabilities[empIdx] * 100

    #retrieve this employee's SHAP values and aggregate by original feature name so that OHE-expanded columns (e.g. OverTime_No, OverTime_Yes) are combined into a single attribution for the original column before ranking
    employeeShapRow = shapDf.iloc[empIdx]
    originalShapAgg = {}
    for expandedName, shapVal in employeeShapRow.items():
        origName = getOriginalFeatureName(expandedName)
        originalShapAgg[origName] = originalShapAgg.get(origName, 0) + abs(shapVal)

    #sort by absolute aggregated SHAP value descending so the most influential features for this specific employee are ranked first
    sortedFeatures = sorted(originalShapAgg.items(), key=lambda x: x[1], reverse=True)
    top3Features = sortedFeatures[:3]

    print(f"Employee {displayRank} (test set row {empIdx})")
    print(f"  Predicted attrition probability: {attritionPct:.1f}%")
    print("  Top 3 influential features (per-employee SHAP) and retention suggestions:")

    for featureRank, (featureName, shapMagnitude) in enumerate(top3Features, start=1):
        suggestion = retentionSuggestions.get(featureName, defaultSuggestion)
        print(f"    {featureRank}. {featureName} (|SHAP|: {shapMagnitude:.4f})")
        print(f"       Suggestion: {suggestion}")
    print()

#compute mean absolute SHAP value per feature across the full test set to rank global feature importance; this is more reliable than impurity-based importance which can inflate the importance of high-cardinality features
meanAbsShap = shapDf.abs().mean(axis=0)

#aggregate mean absolute SHAP by original feature name so OHE columns are combined
originalMeanShap = {}
for expandedName, meanVal in meanAbsShap.items():
    origName = getOriginalFeatureName(expandedName)
    originalMeanShap[origName] = originalMeanShap.get(origName, 0) + meanVal

#sort by aggregated mean absolute SHAP descending for the importance bar chart
rankedShapImportance = sorted(originalMeanShap.items(), key=lambda x: x[1], reverse=True)
top15ShapNames = [name for name, _ in rankedShapImportance[:15]]
top15ShapValues = [val for _, val in rankedShapImportance[:15]]

#plot top 15 features by mean absolute SHAP value as a horizontal bar chart; SHAP-based importance is preferred here because it reflects the actual model output rather than tree-splitting statistics used in impurity-based importance
figImportance, axImportance = plt.subplots(figsize=(10, 7))
yPositions = np.arange(len(top15ShapNames))

axImportance.barh(yPositions, top15ShapValues, color='steelblue', edgecolor='white')
axImportance.set_yticks(yPositions)
axImportance.set_yticklabels(top15ShapNames, fontsize=11)
axImportance.invert_yaxis()
axImportance.set_xlabel('Mean Absolute SHAP Value', fontsize=12)
axImportance.set_title(
    'Top 15 Features by Mean Absolute SHAP Value\nEmployee Attrition Prediction',
    fontsize=12
)
axImportance.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

shapImportancePath = 'v3a_Sonnet_final_shap_importance.png'
plt.savefig(shapImportancePath, dpi=150)
plt.show()
plt.close()
print(f"SHAP importance chart saved to: {shapImportancePath}")

#compute confusion matrix at threshold 0.3 because that is the operational threshold used in the Oracle; showing the matrix at 0.5 would misrepresent real-world performance
testConfusionMatrix = confusion_matrix(yTest, testPredicted03)
print("\nConfusion Matrix at threshold 0.3 (held-out test set):")
print(testConfusionMatrix)

#plot the confusion matrix as a colour-coded heatmap so the balance between false negatives and false positives is immediately visible without reading raw numbers
figCm, axCm = plt.subplots(figsize=(8, 6))
heatmapImage = axCm.imshow(testConfusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(heatmapImage, ax=axCm)

classLabels = ['No Attrition', 'Attrition']
tickPositions = np.arange(len(classLabels))
axCm.set_xticks(tickPositions)
axCm.set_yticks(tickPositions)
axCm.set_xticklabels(classLabels, fontsize=12)
axCm.set_yticklabels(classLabels, fontsize=12)

#use contrasting text colour so cell counts remain legible against both light and dark heatmap cells; white text on dark cells and black text on light cells
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
    'Confusion Matrix at Threshold 0.3 – Calibrated Random Forest\n'
    f'AUC-ROC: {testAucRoc:.3f} | F1: {testF1at03:.3f} | Recall: {testRecallAt03:.3f}',
    fontsize=12
)

plt.tight_layout()

cmPlotPath = 'v3a_Sonnet_final_confusion_matrix.png'
plt.savefig(cmPlotPath, dpi=150)
plt.show()
plt.close()
print(f"Confusion matrix heatmap saved to: {cmPlotPath}")
