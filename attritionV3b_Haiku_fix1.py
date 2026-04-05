import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)

#loadRawDataBeforeAnyTransformationSoFeatureColumnsCanBeValidated
rawData = pd.read_csv('IBM_HR_Analytics_Employee_Attrition_and_Performance.csv')

print("Dataset shape:", rawData.shape)
print("\nColumn names:")
print(rawData.columns.tolist())

#dropColumnsThatCarryNoPredictiveSignal:EmployeeCountAndStandardHoursAreConstants,EmployeeNumberIsAnArbitraryId,Over18HasASingleValueAcrossAllRows
constantColumns = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
rawData = rawData.drop(columns=constantColumns)

#isolateProtectedDemographicAttributesIntoASeparateAuditDataframeBeforeBuildingTheFeatureMatrixSoTheyCannotLeakIntoTheModelCausingIndirectDiscrimination;theseColumnsArePreservedForFairnessAuditsOnly
auditColumns = ['Age', 'Gender', 'MaritalStatus']
auditData = rawData[auditColumns].copy()
print("\nAudit dataframe shape (protected attributes):", auditData.shape)
print("Audit columns:", auditData.columns.tolist())

#encodeTheBinaryTargetAs1Yes=AttritionAnd0No=RetainedBeforeSplittingSoPositiveClassMetricsUnambiguouslyReferToEmployeesWhoLeft
yTarget = (rawData['Attrition'] == 'Yes').astype(int)
print("\nAttrition class distribution:")
print(yTarget.value_counts())

#buildTheFeatureMatrixExcludingTheTargetAndAllAuditColumns;thisGuaranteesThatAgeGenderAndMaritalStatusAreAbsentFromTraining
excludeColumns = ['Attrition'] + auditColumns
featureData = rawData.drop(columns=excludeColumns)

#confirmNoProtectedAttributesRemainInTheFeatureMatrixBeforeAnySplitting
assert not any(col in featureData.columns for col in auditColumns), \
    "Protected attribute found in feature matrix"

#performAn80_20StratifiedSplitBeforeAnyPreprocessingOrModelFittingSoTheHeldOutTestSetIsNeverTouchedUntilFinalEvaluationAndCannotInfluencePreprocessingDecisionsOrHyperparameterChoices
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

#defineNominalCategoricalColumnsThatRequireOneHotEncoderToAvoidImplyingFalseOrdinalRelationshipsBetweenUnorderedCategories
nominalColumns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole']

#overTimeIsBinaryYes_NoAndIsTreatedAsANominalColumnToAvoidEncodingItAsAnArbitraryInteger;oneHotEncoderMakesTheIntentExplicitAndConsistent
nominalColumns.append('OverTime')

#identifyAllRemainingNumericColumns;standardScalerIsAppliedSoThatFeaturesMeasuredOnVeryDifferentScalesContributeEquallyToTheModelAndConvergenceIsConsistent;randomForestDoesNotRequireScalingButItIsRetainedToKeepThePreprocessingPipelineIdenticalToV2b_SonnetPyAsRequired
numericColumns = xTrain.select_dtypes(include=[np.number]).columns.tolist()

print("\nNominal columns (OneHotEncoded):", nominalColumns)
print("Numeric columns (StandardScaled):", numericColumns)

#verifyThatTheNominalAndNumericColumnListsCoverAllFeatureColumnsWithNoOverlap;anyUnaccountedColumnsWouldSilentlyBeDroppedByColumnTransformer
allHandled = set(nominalColumns + numericColumns)
allFeature = set(xTrain.columns)
unhandled = allFeature - allHandled
print("\nUnhandled columns (should be empty):", unhandled)
assert len(unhandled) == 0, f"Columns not assigned to any transformer: {unhandled}"

#buildAColumnTransformerThatApplesOneHotEncoderAndStandardScalerInOneStep;handleUnknown=ignorePreventsErrorsIfACategoryUnseenDuringTrainingAppearsduringCrossValidationFolds;remainder=dropDiscardsAnyStrayColumnsSafely
preprocessor = ColumnTransformer(
    transformers=[
        ('oneHot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominalColumns),
        ('scaler', StandardScaler(), numericColumns)
    ],
    remainder='drop'
)

#randomForestIsChosenOverLogisticRegressionBecauseItCapturesNonLinearFeatureInteractionsEgHighOverTimeCombinedWithLowJobSatisfactionThatALinearModelCannotRepresent;theInitialN_estimators=500IsValidBeforeTuningButWillBeOptimizedByGridSearchCv;classWeight=balancedUpWeightsTheMinorityAttritionClassToCorrectForTheRoughly16PercentPositiveRateWithoutRequiringResampling
randomForestModel = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

#combinePreprocessingAndClassifierIntoASinglePipelineSoThatEncodersAndScalersAreFittedOnlyOnTrainingFoldsAndNeverSeeTestFoldDataPreventingAnyFormOfDataLeakageIntoTheCrossValidationEstimate
modelPipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', randomForestModel)
])

#stratifiedKFoldPreservesTheOriginalClassRatioInEveryFold;criticalWhenThePositiveClassIsRareApproximately16PercentAttritionSoEachFoldGivesARepresentativeEstimate
stratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- GridSearchCV: Hyperparameter Tuning on Training Set ---")

#defineTheParameterGridForGridSearchCv;targetingRecallBecauseCatchingEmployeesWhoWillLeaveIsMoreImportantThanAvoidingFalseAlarms(incorrectlyFlaggingEmployeesWhoWillStay)
parameterGrid = {
    'classifier__n_estimators': [100, 300, 500],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_samples': [0.7, 0.8, None]
}

#gridSearchCvExhaustivelyEvaluatesAllCombinationsOfParametersUsingStratifiedKFoldOnTheTrainingSetOnly;theHeldOutTestSetIsNeverTouchedDuringThisTuningProcess
gridSearch = GridSearchCV(
    modelPipeline,
    parameterGrid,
    cv=stratifiedKFold,
    scoring='recall',
    n_jobs=-1,
    verbose=1
)

#fittingGridSearchCvUsingTheTrainingSetOnly;thisEnsuresTheBestParametersAreSelectedBasedOnlyOnTrainingDataAndNotOnConstraintsFromTheTestSet
gridSearch.fit(xTrain, yTrain)

print("\n--- GridSearchCV Results ---")
print(f"Best parameters found: {gridSearch.best_params_}")
print(f"Best cross-validation recall score: {gridSearch.best_score_:.4f}")

#extractTheBestEstimatorAfterFittingIsCompleted;thisIsTheOptimizedModelToWrapWithProbabilityCalibration
bestModel = gridSearch.best_estimator_

#wrapTheUnedRandomForestInCalibratedClassifierCvToEnsureTheProbabilityOutputsAreMeaningfulAndHonest;a70PercentPredictionShouldReflectAGenuine70PercentLikelihoodRatherThanArbitraryUncalibratedProbabilities
calibrationMethod = 'sigmoid'
calibratedModel = CalibratedClassifierCV(bestModel, method=calibrationMethod, cv=5)
calibratedModel.fit(xTrain, yTrain)

print(f"Calibration method: {calibrationMethod}")
print("Calibrated model fitted on training set")

print("\n--- Threshold Tuning: Evaluating at 0.5 and 0.3 ---")

#generateProbabilisticPredictionsOnTheHeldOutTestSetForBothThresholdsUsingCalibratedModelToEnsureProbabilitiesReflectActualRisk
testProbabilitiesCalibrated = calibratedModel.predict_proba(xTest)[:, 1]

#evaluateOnTheTestSetAtTheThreshold0_5DefaultSklearnThreshold
testPredicted05 = (testProbabilitiesCalibrated >= 0.5).astype(int)
testAucRoc05 = roc_auc_score(yTest, testProbabilitiesCalibrated)
testF105 = f1_score(yTest, testPredicted05, pos_label=1, zero_division=0)
testPrecision05 = precision_score(yTest, testPredicted05, pos_label=1, zero_division=0)
testRecall05 = recall_score(yTest, testPredicted05, pos_label=1, zero_division=0)

#evaluateOnTheTestSetAtTheThreshold0_3LowerThresholdToCaptureMoreAttritionAtTheCostOfFalsePositives;usefulForProactiveRetentionPrograms
testPredicted03 = (testProbabilitiesCalibrated >= 0.3).astype(int)
testAucRoc03 = roc_auc_score(yTest, testProbabilitiesCalibrated)
testF103 = f1_score(yTest, testPredicted03, pos_label=1, zero_division=0)
testPrecision03 = precision_score(yTest, testPredicted03, pos_label=1, zero_division=0)
testRecall03 = recall_score(yTest, testPredicted03, pos_label=1, zero_division=0)

#displayThresholdComparisonTableSideBySidesoHrCanMakeAnInformedChoiceAboutWhichThresholdBestMatchesTheirRiskToleranceAndObjectives
print("\nThreshold Comparison - Held-Out Test Set Metrics (positive class = Attrition Yes)")
print("\n{:<20} {:<15} {:<15}".format("Metric", "Threshold 0.5", "Threshold 0.3"))
print("-" * 50)
print("{:<20} {:<15.4f} {:<15.4f}".format("AUC-ROC", testAucRoc05, testAucRoc03))
print("{:<20} {:<15.4f} {:<15.4f}".format("F1-Score", testF105, testF103))
print("{:<20} {:<15.4f} {:<15.4f}".format("Precision", testPrecision05, testPrecision03))
print("{:<20} {:<15.4f} {:<15.4f}".format("Recall", testRecall05, testRecall03))

print("\n--- Oracle Output: Employee Retention Guidance ---")
print("Using threshold 0.3 for flagging at-risk employees")

#retrieveFeatureImportancesFromTheFittedRandomForestInsideThePipeline;theseAreMeanImpurityDecreaseValuesAcrossAllTreesAndReflectTheActualFittedModelNotLogisticRegressionCoefficients;importantlyWeExtractFromTheBaseEstimatorNotFromTheCalibratedWrapperSinceTheBaseTrainerHasTheFeatureImportancesAttribute
fittedForest = bestModel.named_steps['classifier']

#buildAllFeatureNamesAfterOneHotEncodingExpansion;thePreprocessorOutputsTheseInTheExactOrderTheyAreFittedDuringTraining;weDerivTheseByActuallyFittingThePreprocessorStandalone
preprocessedX = preprocessor.fit_transform(xTrain)
allFeatureNames = []

#startWithNominalColumnsAfterOneHotEncodingExpansion;oneHotEncoderCreatesABinaryColumnForEachSubcategoryWithANamingConventionOfOriginalColName_categoryValue
oneHotCategories = preprocessor.named_transformers_['oneHot'].get_feature_names_out(nominalColumns)
allFeatureNames.extend(oneHotCategories)

#appendNumericColumnsSinceTheyComAfterTheOneHotEncodedColumns
allFeatureNames.extend(numericColumns)

print(f"Total features after preprocessing: {len(allFeatureNames)}")

#buildAShapExplainerForTheTreeBasedModel;treeExplainerIsOptimizedForRandomForestAndComputesSHAPValuesEfficiently;weExtractTheBaseForestFromThePipelineNotTheCalibratedWrapper
explainer = shap.TreeExplainer(fittedForest)

#transformTheTestSetDataUsingTheSamePreprocessorSoThatSHAPValuesAlignWithTheFeaturesTheModelSees
xTestProcessed = preprocessor.transform(xTest)

shapValues = explainer.shap_values(xTestProcessed)
#handleBothBinaryAndMultiClassCases;forBinaryClassificationShapValuesIsAListOfTwoArrays
if isinstance(shapValues, list):
    shapValuesAttrition = shapValues[1]  #positiveClass=Attrition
else:
    shapValuesAttrition = shapValues  #fallbackForSingleArray

#mapOneHotEncodingExpandedFeatureNamesBackToOriginalColumnNamesSoRetentionSuggestionsCanBeLookedUpAgainstHumanReadableFeatureNames;eGOverTime_YesMapsToOverTimeSoTheCorrectSuggestionIsRetrieved
def getOriginalFeatureName(expandedName):
    #stripTheOneHotEncodingSuffixByCheckingIfTheExpandedNameStartsWithAnyNominalColumn
    for nomCol in nominalColumns:
        if expandedName.startswith(nomCol + '_'):
            return nomCol
    return expandedName

#mapOriginalColumnNamesToRetentionSuggestionssoThatActionableGuidanceIsProducedForEachFlaggedEmployeeRatherThanAGenericWarning
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
    'YearsAtCompany': "Recognise loyalty and ensure engagement and development opportunities reflect tenure"
}
defaultSuggestion = "Review this factor with the employee line manager"

#identifyAllTestSetIndicesWhereTheModelPredictAnAttritionProbabilityAbove0_3;resettingTheIndexEnsuresPositionalAlignmentBetweenTestProbabilitiesCalibratedArrayAndTheXTestDataframeRows
xTestReset = xTest.reset_index(drop=True)
flaggedIndices = np.where(testProbabilitiesCalibrated > 0.3)[0]

print(f"\nTotal employees flagged (probability>0.3): {len(flaggedIndices)}")
print(f"Showing first 5 flagged employees:\n")

#iterateOnlyOverTheFirst5FlaggedEmployeesToKeepConsoleOutputReadable;allFlaggedEmployeesCouldBeExportedToAFileForOperationalUse
for displayRank, empIdx in enumerate(flaggedIndices[:5], start=1):
    attritionPct = testProbabilitiesCalibrated[empIdx] * 100
    
    #forThisEmployeeGetTheSHAPValuesAndConvertToAbsoluteValuesToIdentifyTheFeaturesWithHighestImpactRegardlessOfSign
    employeeSHAPValues = shapValuesAttrition[empIdx]
    employeeSHAPAbsolute = np.abs(employeeSHAPValues)
    
    #findTheTop3FeaturesIndicesByAbsoluteSHAPValueForThisSpecificEmployee
    top3Indices = np.argsort(employeeSHAPAbsolute)[-3:][::-1]
    
    print(f"Employee {displayRank} (test set row {empIdx})")
    print(f"  Predicted attrition probability: {attritionPct:.1f}%")
    print("  Top 3 influential features (by SHAP) and retention suggestions:")
    
    for featureRank, featureIdx in enumerate(top3Indices, start=1):
        expandedFeatureName = allFeatureNames[int(featureIdx)]
        originalFeatureName = getOriginalFeatureName(expandedFeatureName)
        shapValue = shapValuesAttrition[empIdx, featureIdx]
        
        suggestion = retentionSuggestions.get(originalFeatureName, defaultSuggestion)
        print(f"    {featureRank}. {originalFeatureName} (SHAP value: {shapValue:.4f})")
        print(f"       Suggestion: {suggestion}")
    print()

#plotTheTop15FeaturesByMeanAbsoluteSHAPValueAsAHorizontalBarChartSoThatHRStakeholdersCanImmediatelyIdentifyWhichFactorsMostInfluenceAttritionRiskWithoutNeedingToInterpretRawCoefficients;meanAbsoluteSHAPValuesGiveATrueGlobalFeatureImportanceMeasureThatAccountsForBothMagnitudeAndFrequencyOfImpact
meanAbsSHAPValues = np.abs(shapValuesAttrition).mean(axis=0)
shapImportanceSeries = pd.Series(meanAbsSHAPValues, index=allFeatureNames).sort_values(ascending=False)

#mapBackToOriginalFeatureNamesAndAggregateOneHotEncodingColumnssoTheyTopPearInTheChartCorrectly
originalFeatureSHAPDict = {}
for featureName, shapValue in shapImportanceSeries.items():
    originalName = getOriginalFeatureName(featureName)
    if originalName not in originalFeatureSHAPDict:
        originalFeatureSHAPDict[originalName] = shapValue
    else:
        originalFeatureSHAPDict[originalName] += shapValue

shapImportanceOriginal = pd.Series(originalFeatureSHAPDict).sort_values(ascending=False)
top15SHAPFeatures = shapImportanceOriginal.head(15)

#createAHorizontalBarChartWithCleanStylingForHRStakeholders
figSHAP, axSHAP = plt.subplots(figsize=(10, 7))
top15SHAPFeatures.plot(kind='barh', ax=axSHAP, color='steelblue', edgecolor='white')
axSHAP.set_xlabel('Mean Absolute SHAP Value', fontsize=12)
axSHAP.set_title(
    'Top 15 Feature Importances – Mean Absolute SHAP Values\nEmployee Attrition Prediction',
    fontsize=12
)
axSHAP.grid(axis='x', linestyle='--', alpha=0.5)
axSHAP.invert_yaxis()
plt.tight_layout()

shapPlotPath = 'v3_sonnet_fix1_shap_importance.png'
plt.savefig(shapPlotPath, dpi=150)
plt.close()
print(f"\nTop 15 SHAP importance chart saved to: {shapPlotPath}")

#computeTheConfusionMatrixOnHeldOutTestSetPredictionsAtTheThreshold0_3ToGiveAnUnbiasedViewOfTheTradeoffBetweenFalseNegativesMissedAttritionAndFalsePositivesUnnecessaryInterventions;rowsAreTrueLabelsColumnsArePredictedLabels
testConfusionMatrix = confusion_matrix(yTest, testPredicted03)
print("\nConfusion Matrix (held-out test set, threshold 0.3):")
print(testConfusionMatrix)

#plotTheConfusionMatrixAsAColorCodedHeatmapSoTheBalanceBetweenFalseNegativesAndFalsePositivesIsImmediatelyVisibleWithoutReadingRawNumbers
figCM, axCM = plt.subplots(figsize=(8, 6))
sns.heatmap(testConfusionMatrix, annot=True, fmt='d', cmap='Blues', cbar=False,
           xticklabels=['No Attrition', 'Attrition'],
           yticklabels=['No Attrition', 'Attrition'],
           ax=axCM, annot_kws={'fontsize': 14, 'fontweight': 'bold'})

axCM.set_xlabel('Predicted Label', fontsize=13)
axCM.set_ylabel('True Label', fontsize=13)
axCM.set_title(
    'Confusion Matrix – Random Forest (Calibrated)\nEmployee Attrition Prediction (Threshold 0.3) '
    f'| AUC-ROC: {testAucRoc03:.3f} | F1: {testF103:.3f}',
    fontsize=12
)

plt.tight_layout()

cmPlotPath = 'v3_sonnet_fix1_confusion_matrix.png'
plt.savefig(cmPlotPath, dpi=150)
plt.close()
print(f"Confusion matrix heatmap saved to: {cmPlotPath}")

print("\n--- Final Model Summary ---")
print(f"Model: Random Forest (Calibrated with {calibrationMethod})")
print(f"Best hyperparameters from GridSearchCV: {gridSearch.best_params_}")
print(f"Threshold used for Oracle output: 0.3")
print(f"Test set AUC-ROC at 0.3: {testAucRoc03:.4f}")
print(f"Test set F1 at 0.3: {testF103:.4f}")
print(f"Test set Recall at 0.3: {testRecall03:.4f}")
print(f"Employees flagged for intervention: {len(flaggedIndices)} / {len(xTest)}")
