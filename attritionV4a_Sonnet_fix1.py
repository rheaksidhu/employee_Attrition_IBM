#attritionV4a_Sonnet_fix1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier
from scipy.stats import pointbiserialr

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

#-----------------------------------------------------------------------
#SECTION 1: CORRELATION ANALYSIS
#run before any modelling to provide model-independent evidence of which features are most associated with attrition; helps justify feature engineering decisions independent of classifier behaviour
#-----------------------------------------------------------------------

print("\n--- Correlation Analysis (model-independent, run before any modelling) ---")

#build a working copy of featureData for correlation purposes only; encode categoricals as integers so point-biserial correlation can be applied uniformly to all columns
correlationData = featureData.copy()

#define the same nominal columns used in the model preprocessing so encoding is consistent
nominalColumnsForCorr = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'OverTime']

#label-encode categorical columns as integers before computing correlation; point-biserial requires numeric inputs and ordinal encoding is sufficient for a marginal correlation screen
labelEnc = LabelEncoder()
for col in nominalColumnsForCorr:
    correlationData[col] = labelEnc.fit_transform(correlationData[col].astype(str))

#calculate point-biserial correlation between each feature and the binary attrition target; point-biserial is the correct choice when one variable is binary and the other is continuous or integer-valued
correlationResults = {}
for col in correlationData.columns:
    corrVal, _ = pointbiserialr(correlationData[col].values, yTarget.values)
    correlationResults[col] = corrVal

#sort features by absolute correlation descending so the most strongly associated features appear first in the chart
sortedCorrelations = sorted(correlationResults.items(), key=lambda x: abs(x[1]), reverse=True)
top15CorrNames = [name for name, _ in sortedCorrelations[:15]]
top15CorrValues = [val for _, val in sortedCorrelations[:15]]

#use colour to distinguish positive and negative correlations so the direction of association is immediately visible; positive correlation means higher feature value is associated with more attrition
barColours = ['tomato' if v > 0 else 'steelblue' for v in top15CorrValues]

figCorr, axCorr = plt.subplots(figsize=(10, 7))
yPosCorr = np.arange(len(top15CorrNames))
axCorr.barh(yPosCorr, top15CorrValues, color=barColours, edgecolor='white')
axCorr.set_yticks(yPosCorr)
axCorr.set_yticklabels(top15CorrNames, fontsize=11)
axCorr.invert_yaxis()
axCorr.axvline(0, color='black', linewidth=0.8)
axCorr.set_xlabel('Point-Biserial Correlation with Attrition', fontsize=12)
axCorr.set_title(
    'Top 15 Features by Absolute Correlation with Attrition\n(Red = positive, Blue = negative)',
    fontsize=12
)
axCorr.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

corrChartPath = 'v4a_Sonnet_final_correlation_chart.png'
plt.savefig(corrChartPath, dpi=150)
plt.show()
plt.close()
print(f"Correlation chart saved to: {corrChartPath}")

#-----------------------------------------------------------------------
#SECTION 2: TRAIN/TEST SPLIT AND PREPROCESSING
#perform split after correlation analysis but before any model fitting so held-out test set is never touched until final evaluation
#-----------------------------------------------------------------------

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

#transform both splits using statistics fitted on xTrain only; xTest is transformed here but will not be evaluated until the final evaluation section to preserve the integrity of the held-out set
xTrainTransformed = preprocessor.transform(xTrain)
xTestTransformed = preprocessor.transform(xTest)

#recover feature names after OHE expansion so SHAP values can be mapped back to human-readable column names for the Oracle output and importance chart
oheFeatureNames = preprocessor.named_transformers_['oneHot'].get_feature_names_out(nominalColumns).tolist()
allFeatureNames = oheFeatureNames + numericColumns

#-----------------------------------------------------------------------
#SECTION 3: XGBOOST CLASSIFIER AND HYPERPARAMETER TUNING
#XGBoost is used instead of RandomForest because it natively handles class imbalance via scale_pos_weight and produces well-calibrated probabilities without requiring a separate calibration wrapper
#-----------------------------------------------------------------------

#StratifiedKFold preserves the original class ratio in every fold; critical when the positive class is rare (~16% attrition) so each fold gives a representative estimate
stratifiedKfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#define the XGBoost-specific hyperparameter search space; these parameters control tree depth, learning rate, subsampling and feature sampling which are the primary levers for XGBoost generalisation; recall is used as the scoring metric because missing an employee who will leave (false negative) is more costly than a false alarm
paramGrid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

#scale_pos_weight=5.2 is the ratio of negative to positive class (1233/237) and corrects for class imbalance natively inside XGBoost without requiring a separate balancing step; eval_metric='logloss' suppresses verbose per-tree output while still monitoring convergence; XGBoost produces well-calibrated probabilities natively so CalibratedClassifierCV is not needed
baseXgb = XGBClassifier(
    scale_pos_weight=5.2,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

#RandomizedSearchCV samples 40 random combinations from the parameter grid rather than exhaustively testing all combinations, finding near-optimal parameters in a fraction of the time while covering a broader search space; fitted only on xTrainTransformed so the test set is never passed here which guarantees the held-out evaluation remains completely unbiased
print("\n--- Running RandomizedSearchCV on training set only (this may take several minutes) ---")
randomSearch = RandomizedSearchCV(
    estimator=baseXgb,
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

#-----------------------------------------------------------------------
#IMPROVEMENT 2: CROSS-VALIDATION AUC ON TRAINING SET
#cross_val_score is run on xTrainTransformed only using the best estimator; this quantifies how consistently the model generalises across different subsets of training data and flags whether the held-out test AUC is a stable or lucky result
#-----------------------------------------------------------------------

#extract best parameters from RandomizedSearchCV before creating the early-stopped model; cross-validation is run here on the raw best estimator to measure training-set generalisation independently of early stopping
bestParamsFromSearch = randomSearch.best_params_

#build a fresh XGBClassifier using the best parameters so cross_val_score operates on the same hyperparameter configuration that RandomizedSearchCV selected; this is run on xTrainTransformed only and xTestTransformed is never touched here
cvXgbEstimator = XGBClassifier(
    **bestParamsFromSearch,
    scale_pos_weight=5.2,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

#cross_val_score uses xTrainTransformed and yTrain only; scoring='roc_auc' is consistent with the final evaluation metric so cross-validation and held-out AUC are directly comparable; the same stratifiedKfold used in RandomizedSearchCV is reused so the fold structure is consistent
cvAucScores = cross_val_score(
    cvXgbEstimator,
    xTrainTransformed,
    yTrain,
    cv=stratifiedKfold,
    scoring='roc_auc',
    n_jobs=-1
)

cvAucMean = cvAucScores.mean()
cvAucStd = cvAucScores.std()

#-----------------------------------------------------------------------
#IMPROVEMENT 3: EARLY STOPPING TO DETERMINE OPTIMAL NUMBER OF TREES
#RandomizedSearchCV selected n_estimators=100 which is the grid floor; early stopping with a 500-tree ceiling allows XGBoost to find the true optimum without being constrained by the search grid minimum
#-----------------------------------------------------------------------

#create an internal validation split from xTrainTransformed only using stratified sampling to preserve class balance; this split is never xTestTransformed which would violate the held-out evaluation guarantee; random_state=42 ensures reproducibility
xTrainEs, xValEs, yTrainEs, yValEs = train_test_split(
    xTrainTransformed, yTrain,
    test_size=0.15,
    stratify=yTrain,
    random_state=42
)

#build the early-stopped model using best parameters but with n_estimators=500 as the upper ceiling; early_stopping_rounds=20 halts training if validation logloss does not improve for 20 consecutive rounds, finding the true optimum rather than stopping at the grid floor of 100
earlyStopXgb = XGBClassifier(
    **{k: v for k, v in bestParamsFromSearch.items() if k != 'n_estimators'},
    n_estimators=500,
    scale_pos_weight=5.2,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=20,
    n_jobs=-1
)

#fit with eval_set pointing to the internal validation split only; xTestTransformed is never passed here so the held-out test set remains untouched until final evaluation
earlyStopXgb.fit(
    xTrainEs, yTrainEs,
    eval_set=[(xValEs, yValEs)],
    verbose=False
)

#retrieve the optimal number of trees identified by early stopping so it can be reported and used as evidence that the grid floor was a binding constraint
optimalNEstimators = earlyStopXgb.best_iteration + 1
print(f"\nEarly stopping optimal number of trees: {optimalNEstimators}")

#refit the final model on the full xTrainTransformed using the optimal n_estimators found by early stopping; refitting on the full training data (not just the early-stopping split) maximises the information available to the model before held-out evaluation
bestXgbModel = XGBClassifier(
    **{k: v for k, v in bestParamsFromSearch.items() if k != 'n_estimators'},
    n_estimators=optimalNEstimators,
    scale_pos_weight=5.2,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
bestXgbModel.fit(xTrainTransformed, yTrain)

#-----------------------------------------------------------------------
#SECTION 4: EVALUATION ON HELD-OUT TEST SET
#test set is evaluated here for the first time; it has never been seen by RandomizedSearchCV, the preprocessor fit step, cross_val_score, or the early stopping validation split
#-----------------------------------------------------------------------

#generate predicted probabilities on the held-out test set; XGBoost produces calibrated probabilities natively so no further calibration step is required
testProbabilities = bestXgbModel.predict_proba(xTestTransformed)[:, 1]

#apply all three thresholds to enable a comparison table; 0.5 is the conventional default, 0.4 is an intermediate operating point, and 0.3 is the operationally preferred threshold that maximises recall at the cost of some precision
testPredicted05 = (testProbabilities >= 0.5).astype(int)
testPredicted04 = (testProbabilities >= 0.4).astype(int)
testPredicted03 = (testProbabilities >= 0.3).astype(int)

#compute metrics separately at each threshold; pos_label=1 is explicit to ensure all metrics refer to the positive (attrition) class only and not the macro or weighted average
testAucRoc = roc_auc_score(yTest, testProbabilities)

testF1at05 = f1_score(yTest, testPredicted05, pos_label=1)
testPrecisionAt05 = precision_score(yTest, testPredicted05, pos_label=1, zero_division=0)
testRecallAt05 = recall_score(yTest, testPredicted05, pos_label=1)

#improvement 1: threshold 0.4 metrics calculated on held-out test set to show precision-recall trade-off at an intermediate operating point between 0.5 and 0.3
testF1at04 = f1_score(yTest, testPredicted04, pos_label=1)
testPrecisionAt04 = precision_score(yTest, testPredicted04, pos_label=1, zero_division=0)
testRecallAt04 = recall_score(yTest, testPredicted04, pos_label=1)

testF1at03 = f1_score(yTest, testPredicted03, pos_label=1)
testPrecisionAt03 = precision_score(yTest, testPredicted03, pos_label=1, zero_division=0)
testRecallAt03 = recall_score(yTest, testPredicted03, pos_label=1)

#print cross-validation AUC alongside held-out test AUC so the two can be compared directly; a small gap confirms the model generalises consistently rather than overfitting to the specific held-out split
print(f"\nCross-validation AUC (5-fold, training set): {cvAucMean:.3f} ± {cvAucStd:.3f}")
print(f"Held-out test AUC: {testAucRoc:.3f}")

print("\n--- Test Set Performance (held-out data, never touched during tuning) ---")
print(f"AUC-ROC (all thresholds): {testAucRoc:.4f}")
print(f"\n{'Metric':<25} {'Threshold 0.5':>15} {'Threshold 0.4':>15} {'Threshold 0.3':>15}")
print("-" * 73)
print(f"{'Precision (positive)':<25} {testPrecisionAt05:>15.4f} {testPrecisionAt04:>15.4f} {testPrecisionAt03:>15.4f}")
print(f"{'Recall (positive)':<25} {testRecallAt05:>15.4f} {testRecallAt04:>15.4f} {testRecallAt03:>15.4f}")
print(f"{'F1-Score (positive)':<25} {testF1at05:>15.4f} {testF1at04:>15.4f} {testF1at03:>15.4f}")

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
    'Confusion Matrix at Threshold 0.3 – XGBoost Classifier\n'
    f'AUC-ROC: {testAucRoc:.3f} | F1: {testF1at03:.3f} | Recall: {testRecallAt03:.3f}',
    fontsize=12
)

plt.tight_layout()

cmPlotPath = 'v4a_Sonnet_final_confusion_matrix.png'
plt.savefig(cmPlotPath, dpi=150)
plt.show()
plt.close()
print(f"Confusion matrix heatmap saved to: {cmPlotPath}")

#-----------------------------------------------------------------------
#SECTION 5: SHAP PER-EMPLOYEE EXPLANATIONS AND ORACLE OUTPUT
#TreeExplainer is used because XGBoost is a tree ensemble; it computes exact Shapley values per employee rather than relying on global feature importances which do not reflect individual-level attribution
#-----------------------------------------------------------------------

#SHAP TreeExplainer is efficient for tree-based models and computes exact Shapley values rather than approximations; per-employee values are required for the Oracle output so global feature importances alone are insufficient
shapExplainer = shap.TreeExplainer(bestXgbModel)
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

#wrap SHAP values in a DataFrame indexed by feature name for easy per-row lookup during Oracle output
shapDf = pd.DataFrame(shapValues, columns=allFeatureNames)

#map OHE-expanded feature names back to their original column name so that retention suggestions can be looked up against human-readable feature names; e.g. 'OverTime_Yes' maps to 'OverTime' so the correct suggestion is retrieved
def getOriginalFeatureName(expandedName):
    #strip the OHE suffix by checking if the expanded name starts with any nominal column so that OverTime_Yes and OverTime_No are both attributed to OverTime
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

    #sort by absolute aggregated SHAP value descending so the most influential features for this specific employee are ranked first; per-employee ranking is critical because global importance does not reflect individual-level attribution
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

#aggregate mean absolute SHAP by original feature name so OHE columns are combined before ranking
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
    'Top 15 Features by Mean Absolute SHAP Value\nEmployee Attrition Prediction (XGBoost)',
    fontsize=12
)
axImportance.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

shapImportancePath = 'v4a_Sonnet_final_shap_importance.png'
plt.savefig(shapImportancePath, dpi=150)
plt.show()
plt.close()
print(f"SHAP importance chart saved to: {shapImportancePath}")

#-----------------------------------------------------------------------
#SECTION 6: FAIRNESS AUDIT
#align test set indices with the audit dataframe to retrieve protected characteristics for each test employee; index alignment is used rather than positional lookup to avoid row mismatches after splitting
#-----------------------------------------------------------------------

print("\n--- Fairness Audit ---")

#use the original xTest index (before any reset) to retrieve the corresponding rows from auditData; this is the correct approach because xTest retains the original row indices from the full dataset after stratified splitting, and auditData shares those same indices
auditTest = auditData.loc[xTest.index].copy()

#reset the audit index to match the positional indices used by testProbabilities and flaggedIndices so that positional lookups are consistent
auditTest = auditTest.reset_index(drop=True)

#define flagged and not-flagged groups using the same 0.3 threshold applied throughout the script; this ensures the fairness audit evaluates the same decisions the Oracle makes
flaggedMask = testProbabilities >= 0.3
notFlaggedMask = ~flaggedMask

auditFlagged = auditTest[flaggedMask]
auditNotFlagged = auditTest[notFlaggedMask]

print(f"Flagged employees: {flaggedMask.sum()}")
print(f"Not flagged employees: {notFlaggedMask.sum()}")

#compute gender and marital status distributions as percentages within each group; percentages rather than counts allow fair comparison when group sizes differ
def getDistributionPct(series):
    #normalise=True returns proportions; multiplying by 100 converts to percentage for readability
    return (series.value_counts(normalize=True) * 100).round(1)

flaggedGender = getDistributionPct(auditFlagged['Gender'])
notFlaggedGender = getDistributionPct(auditNotFlagged['Gender'])

flaggedMarital = getDistributionPct(auditFlagged['MaritalStatus'])
notFlaggedMarital = getDistributionPct(auditNotFlagged['MaritalStatus'])

flaggedMeanAge = auditFlagged['Age'].mean()
notFlaggedMeanAge = auditNotFlagged['Age'].mean()

print("\n-- Gender distribution (%) --")
print(f"{'':>20} {'Flagged':>10} {'Not Flagged':>12}")
for genderLabel in ['Male', 'Female']:
    fVal = flaggedGender.get(genderLabel, 0.0)
    nfVal = notFlaggedGender.get(genderLabel, 0.0)
    print(f"{genderLabel:>20} {fVal:>10.1f} {nfVal:>12.1f}")

print("\n-- Marital status distribution (%) --")
print(f"{'':>20} {'Flagged':>10} {'Not Flagged':>12}")
for statusLabel in ['Single', 'Married', 'Divorced']:
    fVal = flaggedMarital.get(statusLabel, 0.0)
    nfVal = notFlaggedMarital.get(statusLabel, 0.0)
    print(f"{statusLabel:>20} {fVal:>10.1f} {nfVal:>12.1f}")

print("\n-- Mean age --")
print(f"{'Flagged mean age':>20}: {flaggedMeanAge:.1f}")
print(f"{'Not flagged mean age':>20}: {notFlaggedMeanAge:.1f}")

#plot grouped bar chart comparing demographic distributions between flagged and not-flagged groups; a grouped layout makes pairwise comparisons between groups easier to read than stacked bars
allGenderLabels = sorted(set(list(flaggedGender.index) + list(notFlaggedGender.index)))
allMaritalLabels = sorted(set(list(flaggedMarital.index) + list(notFlaggedMarital.index)))

figFairness, axesFairness = plt.subplots(1, 2, figsize=(14, 6))

#gender subplot: plot bars side by side so each gender category shows both groups at once
xGender = np.arange(len(allGenderLabels))
barWidth = 0.35
flaggedGenderVals = [flaggedGender.get(g, 0.0) for g in allGenderLabels]
notFlaggedGenderVals = [notFlaggedGender.get(g, 0.0) for g in allGenderLabels]

axesFairness[0].bar(xGender - barWidth/2, flaggedGenderVals, barWidth, label='Flagged', color='tomato', edgecolor='white')
axesFairness[0].bar(xGender + barWidth/2, notFlaggedGenderVals, barWidth, label='Not Flagged', color='steelblue', edgecolor='white')
axesFairness[0].set_xticks(xGender)
axesFairness[0].set_xticklabels(allGenderLabels, fontsize=11)
axesFairness[0].set_ylabel('Percentage (%)', fontsize=11)
axesFairness[0].set_title('Gender Distribution\nFlagged vs Not Flagged', fontsize=11)
axesFairness[0].legend()
axesFairness[0].grid(axis='y', linestyle='--', alpha=0.5)

#marital status subplot: same grouped approach for three-category comparison
xMarital = np.arange(len(allMaritalLabels))
flaggedMaritalVals = [flaggedMarital.get(m, 0.0) for m in allMaritalLabels]
notFlaggedMaritalVals = [notFlaggedMarital.get(m, 0.0) for m in allMaritalLabels]

axesFairness[1].bar(xMarital - barWidth/2, flaggedMaritalVals, barWidth, label='Flagged', color='tomato', edgecolor='white')
axesFairness[1].bar(xMarital + barWidth/2, notFlaggedMaritalVals, barWidth, label='Not Flagged', color='steelblue', edgecolor='white')
axesFairness[1].set_xticks(xMarital)
axesFairness[1].set_xticklabels(allMaritalLabels, fontsize=11)
axesFairness[1].set_ylabel('Percentage (%)', fontsize=11)
axesFairness[1].set_title('Marital Status Distribution\nFlagged vs Not Flagged', fontsize=11)
axesFairness[1].legend()
axesFairness[1].grid(axis='y', linestyle='--', alpha=0.5)

figFairness.suptitle(
    f'Fairness Audit – Demographic Comparison at Threshold 0.3\n'
    f'Mean Age: Flagged = {flaggedMeanAge:.1f}, Not Flagged = {notFlaggedMeanAge:.1f}',
    fontsize=12
)
plt.tight_layout()

fairnessPlotPath = 'v4a_Sonnet_final_fairness_audit.png'
plt.savefig(fairnessPlotPath, dpi=150)
plt.show()
plt.close()
print(f"\nFairness audit chart saved to: {fairnessPlotPath}")
