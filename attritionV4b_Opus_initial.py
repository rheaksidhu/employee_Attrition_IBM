#attritionV4b_Opus_initial

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from scipy.stats import pointbiserialr
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier

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

# ──────────────────────────────────────────────────────────
#CORRELATION ANALYSIS: run before any modelling to provide model-independent evidence of which features are most associated with attrition
# ──────────────────────────────────────────────────────────

print("\n--- Correlation Analysis (point-biserial) ---")

#build a temporary copy of features for correlation so the real feature matrix stays untouched
correlationFrame = featureData.copy()

#encode any remaining categorical columns as integers so point-biserial correlation can be computed; this is only for the correlation analysis and does not affect the modelling pipeline
categoricalCorrelationColumns = correlationFrame.select_dtypes(include=['object']).columns.tolist()
labelEncoders = {}
for catCol in categoricalCorrelationColumns:
    labelEncoders[catCol] = LabelEncoder()
    correlationFrame[catCol] = labelEncoders[catCol].fit_transform(correlationFrame[catCol])

#calculate point-biserial correlation between each feature and the binary attrition target; point-biserial is the correct measure when one variable is continuous and the other is dichotomous
correlationResults = {}
for colName in correlationFrame.columns:
    corrCoeff, pValue = pointbiserialr(yTarget, correlationFrame[colName])
    correlationResults[colName] = corrCoeff

#sort by absolute correlation descending to rank features by strength of linear association with attrition
correlationSeries = pd.Series(correlationResults)
topCorrelations = correlationSeries.reindex(correlationSeries.abs().sort_values(ascending=False).index)[:15]

print("Top 15 features by absolute point-biserial correlation with attrition:")
for featName, corrVal in topCorrelations.items():
    print(f"  {featName}: {corrVal:.4f}")

#plot horizontal bar chart of top 15 correlations; colour indicates direction of association (positive = increases attrition risk, negative = protective)
figCorr, axCorr = plt.subplots(figsize=(10, 7))
barColours = ['#e74c3c' if v > 0 else '#2ecc71' for v in topCorrelations.values]
axCorr.barh(range(len(topCorrelations)), topCorrelations.values, color=barColours, edgecolor='white')
axCorr.set_yticks(range(len(topCorrelations)))
axCorr.set_yticklabels(topCorrelations.index, fontsize=11)
axCorr.invert_yaxis()
axCorr.set_xlabel('Point-Biserial Correlation with Attrition', fontsize=12)
axCorr.set_title(
    'Top 15 Features by Absolute Correlation with Attrition\n(Red = positive association, Green = negative association)',
    fontsize=12
)
axCorr.axvline(x=0, color='black', linewidth=0.8)
axCorr.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

correlationChartPath = 'v4b_Opus_initial_correlation_chart.png'
plt.savefig(correlationChartPath, dpi=150)
plt.close()
print(f"Correlation chart saved to: {correlationChartPath}")

# ──────────────────────────────────────────────────────────
#MODELLING: preprocessing, XGBoost training, and hyperparameter tuning
# ──────────────────────────────────────────────────────────

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

#transform both splits using statistics fitted on xTrain only; this mirrors what a Pipeline would do but allows us to pass a plain numpy array to RandomizedSearchCV with a bare XGBClassifier
xTrainTransformed = preprocessor.transform(xTrain)
xTestTransformed = preprocessor.transform(xTest)

#recover feature names after OHE expansion so SHAP values can be mapped back to human-readable column names for the Oracle output and importance chart
oheFeatureNames = preprocessor.named_transformers_['oneHot'].get_feature_names_out(nominalColumns).tolist()
allFeatureNames = oheFeatureNames + numericColumns

#StratifiedKFold preserves the original class ratio in every fold; critical when the positive class is rare (~16% attrition) so each fold gives a representative estimate
stratifiedKfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#define the hyperparameter search space for XGBoost; recall is used as the scoring metric because missing an employee who will leave (false negative) is more costly than a false alarm
paramGrid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

#XGBClassifier with scale_pos_weight=5.2 handles class imbalance natively by weighting the positive class proportionally to the negative-to-positive ratio (1233/237≈5.2); eval_metric='logloss' suppresses verbose output during training; random_state=42 ensures reproducibility
baseXgb = XGBClassifier(
    scale_pos_weight=5.2,
    random_state=42,
    eval_metric='logloss'
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

#extract the best XGBoost model refit on the full training set using the best parameters found during random search; no CalibratedClassifierCV is needed because XGBoost produces well-calibrated probabilities natively via logistic loss
bestXgb = randomSearch.best_estimator_

#make predictions on the held-out test set for evaluation; test set has never been seen by RandomizedSearchCV or the XGBoost model during training
testProbabilities = bestXgb.predict_proba(xTestTransformed)[:, 1]

#apply both 0.5 and 0.3 thresholds for comparison; 0.3 is the operational threshold chosen to optimise recall because missing an employee who will leave is more costly than alerting the business to a false positive
testPredicted05 = (testProbabilities >= 0.5).astype(int)
testPredicted03 = (testProbabilities >= 0.3).astype(int)

#compute evaluation metrics on the positive class only as specified; recall is prioritised because missing an employee who will leave is more costly than a false alarm
testAucRoc = roc_auc_score(yTest, testProbabilities)

testF1at05 = f1_score(yTest, testPredicted05, pos_label=1)
testPrecisionAt05 = precision_score(yTest, testPredicted05, pos_label=1)
testRecallAt05 = recall_score(yTest, testPredicted05, pos_label=1)

testF1at03 = f1_score(yTest, testPredicted03, pos_label=1)
testPrecisionAt03 = precision_score(yTest, testPredicted03, pos_label=1)
testRecallAt03 = recall_score(yTest, testPredicted03, pos_label=1)

print("\n--- Test Set Performance (held-out data, never touched during tuning) ---")
print(f"AUC-ROC (positive class, all thresholds): {testAucRoc:.4f}")

print("\nThreshold Comparison (positive class only):")
print(f"{'Metric':<12} {'Threshold 0.5':>14} {'Threshold 0.3':>14}")
print(f"{'Precision':<12} {testPrecisionAt05:>14.4f} {testPrecisionAt03:>14.4f}")
print(f"{'Recall':<12} {testRecallAt05:>14.4f} {testRecallAt03:>14.4f}")
print(f"{'F1-Score':<12} {testF1at05:>14.4f} {testF1at03:>14.4f}")

# ──────────────────────────────────────────────────────────
#SHAP ANALYSIS: per-employee explanations using TreeExplainer
# ──────────────────────────────────────────────────────────

#SHAP TreeExplainer computes exact Shapley values for tree-based models; this quantifies the contribution of each feature to the model's prediction for each individual employee rather than using global feature importances
shapExplainer = shap.TreeExplainer(bestXgb)
rawShapValues = shapExplainer.shap_values(xTestTransformed)

#handle both legacy and modern SHAP output formats because shap library versions may differ
if isinstance(rawShapValues, list):
    #legacy shap behaviour: list of [class0_array, class1_array]
    shapValues = rawShapValues[1]
elif rawShapValues.ndim == 3:
    #modern shap behaviour: single 3D array where axis 2 is the class dimension
    shapValues = rawShapValues[:, :, 1]
else:
    #XGBoost binary classifier with TreeExplainer typically returns a 2D array directly for the positive class
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
    'Top 15 Features by Mean Absolute SHAP Value\nEmployee Attrition Prediction (XGBoost)',
    fontsize=12
)
axImportance.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

shapImportancePath = 'v4b_Opus_initial_shap_importance.png'
plt.savefig(shapImportancePath, dpi=150)
plt.close()
print(f"SHAP importance chart saved to: {shapImportancePath}")

# ──────────────────────────────────────────────────────────
#CONFUSION MATRIX: visualise classification performance at the operational threshold
# ──────────────────────────────────────────────────────────

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
    'Confusion Matrix at Threshold 0.3 – XGBoost\n'
    f'AUC-ROC: {testAucRoc:.3f} | F1: {testF1at03:.3f} | Recall: {testRecallAt03:.3f}',
    fontsize=12
)

plt.tight_layout()

cmPlotPath = 'v4b_Opus_initial_confusion_matrix.png'
plt.savefig(cmPlotPath, dpi=150)
plt.close()
print(f"Confusion matrix heatmap saved to: {cmPlotPath}")

# ──────────────────────────────────────────────────────────
#FAIRNESS AUDIT: check for demographic bias in flagging decisions
# ──────────────────────────────────────────────────────────

print("\n--- Fairness Audit ---")

#align audit dataframe rows with the test set using the original index from the train_test_split; this ensures each test employee's protected characteristics are correctly matched to their prediction without positional errors
testAuditData = auditData.loc[xTest.index].reset_index(drop=True)

#split test employees into flagged and not-flagged groups based on the 0.3 operational threshold so demographic distributions can be compared between those the model targets for intervention and those it does not
flaggedMask = testProbabilities >= 0.3
flaggedAudit = testAuditData[flaggedMask]
notFlaggedAudit = testAuditData[~flaggedMask]

print(f"Flagged employees: {len(flaggedAudit)}")
print(f"Not flagged employees: {len(notFlaggedAudit)}")

#calculate gender distribution as percentages within each group to detect whether the model disproportionately flags one gender over the other
flaggedGenderPct = flaggedAudit['Gender'].value_counts(normalize=True) * 100
notFlaggedGenderPct = notFlaggedAudit['Gender'].value_counts(normalize=True) * 100

#calculate marital status distribution as percentages to detect whether single, married, or divorced employees are disproportionately flagged
flaggedMaritalPct = flaggedAudit['MaritalStatus'].value_counts(normalize=True) * 100
notFlaggedMaritalPct = notFlaggedAudit['MaritalStatus'].value_counts(normalize=True) * 100

#calculate mean age to detect whether the model systematically flags younger or older employees
flaggedMeanAge = flaggedAudit['Age'].mean()
notFlaggedMeanAge = notFlaggedAudit['Age'].mean()

print("\nGender Distribution (%):")
print(f"  {'Category':<12} {'Flagged':>10} {'Not Flagged':>12}")
for gender in ['Male', 'Female']:
    flaggedVal = flaggedGenderPct.get(gender, 0)
    notFlaggedVal = notFlaggedGenderPct.get(gender, 0)
    print(f"  {gender:<12} {flaggedVal:>10.1f} {notFlaggedVal:>12.1f}")

print("\nMarital Status Distribution (%):")
print(f"  {'Category':<12} {'Flagged':>10} {'Not Flagged':>12}")
for status in ['Single', 'Married', 'Divorced']:
    flaggedVal = flaggedMaritalPct.get(status, 0)
    notFlaggedVal = notFlaggedMaritalPct.get(status, 0)
    print(f"  {status:<12} {flaggedVal:>10.1f} {notFlaggedVal:>12.1f}")

print(f"\nMean Age:")
print(f"  Flagged:     {flaggedMeanAge:.1f}")
print(f"  Not Flagged: {notFlaggedMeanAge:.1f}")

#plot grouped bar chart comparing demographic distributions between flagged and not-flagged groups; this visualisation makes disparities immediately apparent to a non-technical stakeholder reviewing the fairness report
figFairness, axesFairness = plt.subplots(1, 3, figsize=(16, 5))

#gender comparison subplot
genderCategories = ['Male', 'Female']
genderFlaggedVals = [flaggedGenderPct.get(g, 0) for g in genderCategories]
genderNotFlaggedVals = [notFlaggedGenderPct.get(g, 0) for g in genderCategories]
barWidth = 0.35
genderXPositions = np.arange(len(genderCategories))
axesFairness[0].bar(genderXPositions - barWidth/2, genderFlaggedVals, barWidth, label='Flagged', color='#e74c3c')
axesFairness[0].bar(genderXPositions + barWidth/2, genderNotFlaggedVals, barWidth, label='Not Flagged', color='#3498db')
axesFairness[0].set_xlabel('Gender', fontsize=11)
axesFairness[0].set_ylabel('Percentage (%)', fontsize=11)
axesFairness[0].set_title('Gender Distribution', fontsize=12)
axesFairness[0].set_xticks(genderXPositions)
axesFairness[0].set_xticklabels(genderCategories, fontsize=10)
axesFairness[0].legend(fontsize=9)
axesFairness[0].set_ylim(0, 100)

#marital status comparison subplot
maritalCategories = ['Single', 'Married', 'Divorced']
maritalFlaggedVals = [flaggedMaritalPct.get(s, 0) for s in maritalCategories]
maritalNotFlaggedVals = [notFlaggedMaritalPct.get(s, 0) for s in maritalCategories]
maritalXPositions = np.arange(len(maritalCategories))
axesFairness[1].bar(maritalXPositions - barWidth/2, maritalFlaggedVals, barWidth, label='Flagged', color='#e74c3c')
axesFairness[1].bar(maritalXPositions + barWidth/2, maritalNotFlaggedVals, barWidth, label='Not Flagged', color='#3498db')
axesFairness[1].set_xlabel('Marital Status', fontsize=11)
axesFairness[1].set_ylabel('Percentage (%)', fontsize=11)
axesFairness[1].set_title('Marital Status Distribution', fontsize=12)
axesFairness[1].set_xticks(maritalXPositions)
axesFairness[1].set_xticklabels(maritalCategories, fontsize=10)
axesFairness[1].legend(fontsize=9)
axesFairness[1].set_ylim(0, 100)

#mean age comparison subplot; bar chart with only two bars makes the age difference between groups visually clear
ageGroups = ['Flagged', 'Not Flagged']
ageValues = [flaggedMeanAge, notFlaggedMeanAge]
ageColours = ['#e74c3c', '#3498db']
axesFairness[2].bar(ageGroups, ageValues, color=ageColours, width=0.5)
axesFairness[2].set_ylabel('Mean Age', fontsize=11)
axesFairness[2].set_title('Mean Age Comparison', fontsize=12)
#set y-axis floor to 25 so the visual difference is not exaggerated by starting at zero when both means are likely above 30
axesFairness[2].set_ylim(25, max(ageValues) + 5)
for idx, val in enumerate(ageValues):
    axesFairness[2].text(idx, val + 0.3, f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Fairness Audit: Flagged vs Not Flagged Employees', fontsize=14, fontweight='bold')
plt.tight_layout()

fairnessPlotPath = 'v4b_Opus_initial_fairness_audit.png'
plt.savefig(fairnessPlotPath, dpi=150)
plt.close()
print(f"Fairness audit chart saved to: {fairnessPlotPath}")

print("\n--- Script complete ---")
