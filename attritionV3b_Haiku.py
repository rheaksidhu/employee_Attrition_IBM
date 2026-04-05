import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)

#loadRawDataBeforeAnyTransformationSoColumnNamesCanBeVerified
rawData = pd.read_csv('IBM_HR_Analytics_Employee_Attrition_and_Performance.csv')

print("Dataset shape:", rawData.shape)
print("\nColumn names:")
print(rawData.columns.tolist())

#dropColumnsThatCarryNoPredictiveSignal:EmployeeCountAndStandardHoursAreConstants,
#EmployeeNumberIsAnArbitraryID,Over18HasASingleValueAcrossAllRows
constantColumns = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
rawData = rawData.drop(columns=constantColumns)

#isolateProtectedDemographicAttributesIntoASeparateAuditDataframeBeforeBuildingThe
#FeatureMatrixSoTheyCannotLeakIntoTheModelAndCauseIndirectDiscrimination;theseColumns
#ArePreservedForFairnessAuditsOnly
auditColumns = ['Age', 'Gender', 'MaritalStatus']
auditData = rawData[auditColumns].copy()
print("\nAudit dataframe shape (protected attributes):", auditData.shape)
print("Audit columns:", auditData.columns.tolist())

#encodeTheBinaryTargetAs1(Yes=Attrition)And0(No=Retained)BeforeSplittingSoThat
#PositiveClassMetricsUnambiguouslyReferToEmployeesWhoLeft
yTarget = (rawData['Attrition'] == 'Yes').astype(int)
print("\nAttrition class distribution (before train/test split):")
print(yTarget.value_counts())

#buildTheFeatureMatrixExcludingTheTargetAndAllAuditColumns;thisGuaranteeThatAge,
#GenderAndMaritalStatusAreAbsentFromTraining
excludeColumns = ['Attrition'] + auditColumns
featureData = rawData.drop(columns=excludeColumns)

#confirmNoProtectedAttributesRemainInTheFeatureMatrix
assert not any(col in featureData.columns for col in auditColumns), \
    "Protected attribute found in feature matrix"

#stratifiedTrainTestSplit:80%TrainAnd20%TestPreservingClassRatio;theTestSetIsHeld
#OutAndNeverTouchedUntilFinalEvaluation
xTrain, xTest, yTrain, yTest = train_test_split(
    featureData, yTarget,
    test_size=0.2, stratify=yTarget, random_state=42
)

print("\nTrain set shape:", xTrain.shape)
print("Test set shape:", xTest.shape)
print("Train set attrition distribution:")
print(yTrain.value_counts())
print("Test set attrition distribution:")
print(yTest.value_counts())

#defineNominalCategoricalColumnsThatRequireOneHotEncoderToAvoidImplyingAFalseOrdinal
#RelationshipBetweenUnorderedCategories
nominalColumns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole']

#OverTimeIsBinaryYes/NoAndIsTreatedAsANominalColumnToAvoidEncodingItAsAnArbitrary
#Integer;OneHotEncoderMakesTheIntentExplicitAndConsistentWithTheOtherNominals
nominalColumns.append('OverTime')

#identifyAllRemainingNumericColumns;StandardScalerIsAppliedSoThatFeaturesMeasuredOn
#VeryDifferentScales(e.g.MonthlyIncomeVsJobLevel)ContributeEquallyToTheRandomForest
#AndConvergenceIsFaster
numericColumns = xTrain.select_dtypes(include=[np.number]).columns.tolist()

print("\nNominal columns (OneHotEncoded):", nominalColumns)
print("Numeric columns (StandardScaled):", numericColumns)

#verifyThatTheNominalAndNumericColumnListsCoverAllFeatureColumnsWithNoOverlap;any
#UnaccountedColumnsWouldSilentlyBeDroppedByColumnTransformer
allHandled = set(nominalColumns + numericColumns)
allFeature = set(xTrain.columns)
unhandled = allFeature - allHandled
print("\nUnhandled columns (should be empty):", unhandled)
assert len(unhandled) == 0, f"Columns not assigned to any transformer: {unhandled}"

#buildAColumnTransformerThatAppliesOneHotEncoderAndStandardScalerInOneStep;
#handleUnknown='ignore'PreventsErrorsIfACategoryUnseenDuringTrainingAppearsDuring
#CrossValidationFolds;remainder='drop'DiscardsAnyStrayColumnsSafely
preprocessor = ColumnTransformer(
    transformers=[
        ('oneHot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominalColumns),
        ('scaler', StandardScaler(), numericColumns)
    ],
    remainder='drop'
)

#classWeight='balanced'InstructsTheRandomForestToUpWeightTheMinorityClass(Attrition=Yes)
#InverselyProportionalToItsFrequency;thisPreventTheModelFromSimplyPredictingNoFor
#EverySampleToAchieveHighAccuracyOnAnImbalancedDataset;nEstimators=500Provides
#AStableEnsembleWithSufficientNumberOfTreesForReliableFeatureImportanceEstimates
randomForestModel = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

#combinePreprocessingAndModelIntoASinglePipelineSoThatTheScalerAndEncoderAreFitted
#OnlyOnTrainingDataAndNeverSeeTestFoldDataOrHeldOutTestData,PreventingDataLeakage
modelPipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', randomForestModel)
])

#StratifiedKFoldPreservesTheOriginalClassRatioInEveryFold;thisCriticalWhenOneClassIs
#Rare(roughly16%AttritionHere)SoEachFoldIsRepresentative
stratifiedKfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#crossValPredictWithMethod='predictProba'ReturnsOutOfFoldProbabilityEstimatesForEvery
#SampleInTheTrainingSet;usingProbabilitiesRatherThanHardLabelsAllowsAUCROCToBe
#ComputedOnTheFullProbabilityRangeRatherThanASingleThreshold
yProbabilitiesTrain = cross_val_predict(
    modelPipeline, xTrain, yTrain,
    cv=stratifiedKfold, method='predict_proba'
)[:, 1]

#deriveHardBinaryPredictionsAtThreshold0.5ForPrecision,RecallAndF1OnTheTrainingSet
yPredictedTrain = (yProbabilitiesTrain >= 0.5).astype(int)

#evaluateExclusivelyOnThePositiveClass(Attrition=Yes,label=1)UsingCrossValidation
#PredictionsOnTheTrainingSet;accuracyIsOmittedBecauseItIsMisleadingUnderClassImbalance
#AndDoesNotCaptureHowWellTheModelIdentifiesTheMinorityClassThatIsOperationallyMost
#Important
aucRocTrain = roc_auc_score(yTrain, yProbabilitiesTrain)
f1Train = f1_score(yTrain, yPredictedTrain, pos_label=1, zero_division=0)
precisionTrain = precision_score(yTrain, yPredictedTrain, pos_label=1, zero_division=0)
recallTrain = recall_score(yTrain, yPredictedTrain, pos_label=1, zero_division=0)

print("\n--- Cross-Validation Metrics on Training Set (positive class = Attrition Yes) ---")
print(f"AUC-ROC:   {aucRocTrain:.4f}")
print(f"F1-Score:  {f1Train:.4f}")
print(f"Precision: {precisionTrain:.4f}")
print(f"Recall:    {recallTrain:.4f}")

#computeTheConfusionMatrixUsingOutOfFoldPredictionsFromTheTrainingSetToGiveAn
#UnbiasedEstimateOfGeneralisationPerformance;rowsAreTrueLabelsColumnsArePredicted
confusionMatrixTrain = confusion_matrix(yTrain, yPredictedTrain)
print("\nConfusion Matrix (Training Set):")
print(confusionMatrixTrain)

#trainTheFinalModelOnTheFullTrainingSetBeforeFinalEvaluation;thePipelineIsRefit
#ToLearnOnAllAvailableTrainingData
print("\n--- Training Final Model on Full Training Set ---")
modelPipeline.fit(xTrain, yTrain)

#evaluateTheFinalModelOnTheHeldOutTestSetWhichHasNeverBeenTouchedDuringTrainingOr
#CrossValidation;thisGivesAnUnbiasedEstimateOfRealWorldPerformanceOnUnseen Data
yProbabilitiesTest = modelPipeline.predict_proba(xTest)[:, 1]
yPredictedTest = (yProbabilitiesTest >= 0.5).astype(int)

aucRocTest = roc_auc_score(yTest, yProbabilitiesTest)
f1Test = f1_score(yTest, yPredictedTest, pos_label=1, zero_division=0)
precisionTest = precision_score(yTest, yPredictedTest, pos_label=1, zero_division=0)
recallTest = recall_score(yTest, yPredictedTest, pos_label=1, zero_division=0)

print("\n--- Final Evaluation Metrics on Held-Out Test Set (positive class = Attrition Yes) ---")
print(f"AUC-ROC:   {aucRocTest:.4f}")
print(f"F1-Score:  {f1Test:.4f}")
print(f"Precision: {precisionTest:.4f}")
print(f"Recall:    {recallTest:.4f}")

#computeTheConfusionMatrixOnTheHeldOutTestSetToGiveAnUnbiasedEstimateOfGeneralisation
#Performance;rowsAreTrueLabelsColumnsArePredicted
confusionMatrixTest = confusion_matrix(yTest, yPredictedTest)
print("\nConfusion Matrix (Held-Out Test Set):")
print(confusionMatrixTest)

#oracleRefinement:ExtractTopFeaturesFromRandomForestFeatureImportances(NotLogistic
#RegressionCoefficients)AndGenerateRetentionSuggestionsForEmployeesWithPredicted
#AttritionProbabilityAbove0.5
flaggedEmployees = []
for idx, (index, testRow) in enumerate(xTest.iterrows()):
    if yProbabilitiesTest[idx] > 0.5:
        flaggedEmployees.append({
            'testIndex': idx,
            'originalIndex': index,
            'probability': yProbabilitiesTest[idx]
        })

#sortFlaggedEmployeesByAttritionProbabilityInDescendingOrderToIdentifyHighestRisk
flaggedEmployees.sort(key=lambda x: x['probability'], reverse=True)

#mappingFromFeatureNamesToRetentionSuggestionsBasedOnDomainKnowledge;anyUnmappedFeature
#DefaultsToAGenericSuggestion
retentionMapping = {
    'OverTime': 'Consider offering flexible working arrangements or reviewing workload distribution',
    'MonthlyIncome': 'Review compensation against market benchmarks and consider a salary adjustment',
    'JobSatisfaction': 'Conduct a one-to-one satisfaction review and identify specific pain points',
    'YearsSinceLastPromotion': 'Discuss career progression pathway and set clear promotion criteria',
    'WorkLifeBalance': 'Explore flexible working options or additional leave entitlements',
    'Age': 'This is a protected characteristic and should not be used for intervention targeting',
    'DistanceFromHome': 'Consider remote working arrangements or travel support',
    'TotalWorkingYears': 'Recognise experience and ensure role complexity matches seniority',
    'JobLevel': 'Review whether current role reflects the employee\'s skills and experience'
}

#extractFeatureImportancesFromTheFittedRandomForestClassifier;thePreprocessor
#TransformsColumnNamesViaOneHotEncodingAndStandardScalingsoWeNeedToMapBackToOriginal
#FeatureNamesForInterpretability
featureImportances = modelPipeline.named_steps['classifier'].feature_importances_
preprocessorTransformer = modelPipeline.named_steps['preprocessor']

#mapEncodedFeatureIndicesToOriginalColumnNames;OneHotEncodedNominalColumnsExpandTo
#MultipleOneHotColumns(eachCategoryBecomesOneColumn)WhileNumericColumnsArePassedThrough
#AsIs
featureNames = []
featureIdx = 0

#addOneHotEncodedNominalColumnNames;OneHotEncoderOutputsCategoryNamesAsStrings
for transformerName, transformer, columns in preprocessorTransformer.transformers_:
    if transformerName == 'oneHot':
        oneHotEncoder = transformer
        categoriesPerFeature = oneHotEncoder.categories_
        for featureIdx, feature in enumerate(columns):
            for category in categoriesPerFeature[featureIdx]:
                featureNames.append(f"{feature}_{category}")
    elif transformerName == 'scaler':
        for feature in columns:
            featureNames.append(feature)

#matchLengthOfFeatureNamesAndImportances;ifMismatchFoundDebugByPrintingInfo
if len(featureNames) != len(featureImportances):
    print(f"\nWarning: Feature names length ({len(featureNames)}) does not match importances length ({len(featureImportances)})")
    print(f"Feature names: {featureNames[:10]}...")
    print(f"Feature importances shape: {featureImportances.shape}")

#mapFeatureImportancesBackToOriginalColumnNamesForInterpretability;sumImportancesOf
#AllOneHotEncodedCategoriesForEachNominalColumnToGetSingleImportancePerOriginalFeature
importanceByOriginalFeature = {}
for featureIdx, (featureName, importance) in enumerate(zip(featureNames, featureImportances)):
    if '_' in featureName and any(featureName.startswith(nom) for nom in nominalColumns):
        #thisisAOneHotEncodedCategorysoExtractTheOriginalColumnName
        originalFeatureName = featureName.split('_')[0]
        if originalFeatureName not in importanceByOriginalFeature:
            importanceByOriginalFeature[originalFeatureName] = 0.0
        importanceByOriginalFeature[originalFeatureName] += importance
    else:
        #thisIsAnumericColumnOrAlreadyASimpleName
        importanceByOriginalFeature[featureName] = importance

#sortFeaturesByImportanceInDescendingOrderToIdentifyTopContributors
sortedFeatures = sorted(importanceByOriginalFeature.items(), key=lambda x: x[1], reverse=True)

print("\n--- Oracle Refinement: Top 15 Features by Importance ---")
for rank, (featureName, importance) in enumerate(sortedFeatures[:15], 1):
    print(f"{rank}. {featureName}: {importance:.4f}")

#printOracleOutputForTheFirst5FlaggedEmployeesOnlyToKeepTheOutputReadable
print("\n--- Oracle Output: Top 5 Flagged Employees (Predicted to Attrit) ---\n")
for employeeNum, employee in enumerate(flaggedEmployees[:5], 1):
    testIdx = employee['testIndex']
    originalIdx = employee['originalIndex']
    attritionProb = employee['probability']
    
    print(f"Employee #{employeeNum} (Dataset Index: {originalIdx})")
    print(f"Predicted Attrition Probability: {attritionProb*100:.2f}%")
    
    #extractTheTop3OriginalFeaturesContributingToThePredictionForThisSpecificEmployee;
    #RandomForestFeatureImportancesAreGlobalAndDonNotVaryPerSample,SoWeUseGlobalTop3
    #Features;ifMoreSampleLevelExplanationIsNeededInterpretableMLMethodsLikeSHAPWould
    #BeRequired
    top3Features = sortedFeatures[:3]
    
    print("Top 3 Contributing Features:")
    for featureRank, (featureName, importance) in enumerate(top3Features, 1):
        suggestion = retentionMapping.get(featureName, 'Review this factor with the employee\'s line manager')
        print(f"  {featureRank}. {featureName} (Importance: {importance:.4f})")
        print(f"     Suggestion: {suggestion}")
    
    print()

#plotAndSaveTheRandomForestFeatureImportanceAsAHorizontalBarChartShowingTheTop15
#Features;horizontalLayoutMakesFeatureNamesMoreReadableThanVerticalBars
top15Features = sortedFeatures[:15]
featureNamesTop15 = [f[0] for f in top15Features]
importancesTop15 = [f[1] for f in top15Features]

figureFeature, axisFeature = plt.subplots(figsize=(10, 8))
axisFeature.barh(featureNamesTop15, importancesTop15, color='steelblue')
axisFeature.set_xlabel('Feature Importance', fontsize=12)
axisFeature.set_ylabel('Feature', fontsize=12)
axisFeature.set_title('Random Forest Feature Importance (Top 15 Features)', fontsize=13)
axisFeature.invert_yaxis()
plt.tight_layout()

#saveTheFeatureImportancePlotAsPNGSoItCanBeEmbeddedInTheAssessmentReport
featureOutputFilePath = 'v3_haiku_rf_feature_importance.png'
plt.savefig(featureOutputFilePath, dpi=150)
plt.show()
plt.close()

print(f"Feature importance plot saved to: {featureOutputFilePath}")

#plotAndSaveTheConfusionMatrixHeatmapOfTheHeldOutTestSetResults;showsTheBalanceBetween
#FalseNegatives(missedAttrition)AndFalsePositives(unnecessaryIntervention)
figureConfusion, axisConfusion = plt.subplots(figsize=(8, 6))

heatmapImageTest = axisConfusion.imshow(confusionMatrixTest, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(heatmapImageTest, ax=axisConfusion)

classLabels = ['No Attrition', 'Attrition']
tickPositions = np.arange(len(classLabels))
axisConfusion.set_xticks(tickPositions)
axisConfusion.set_yticks(tickPositions)
axisConfusion.set_xticklabels(classLabels, fontsize=12)
axisConfusion.set_yticklabels(classLabels, fontsize=12)

#useContrastingTextColourSoCellCountsRemainLegibleAgainstBothLightAndDarkHeatmapCells;
#whiteTextOnDarkCellsAndBlackTextOnLightCells
thresholdValue = confusionMatrixTest.max() / 2.0
for rowIndex in range(confusionMatrixTest.shape[0]):
    for colIndex in range(confusionMatrixTest.shape[1]):
        cellValue = confusionMatrixTest[rowIndex, colIndex]
        textColour = 'white' if cellValue > thresholdValue else 'black'
        axisConfusion.text(
            colIndex, rowIndex, str(cellValue),
            ha='center', va='center',
            color=textColour, fontsize=14, fontweight='bold'
        )

axisConfusion.set_xlabel('Predicted Label', fontsize=13)
axisConfusion.set_ylabel('True Label', fontsize=13)
axisConfusion.set_title(
    'Confusion Matrix – Random Forest (Balanced)\nEmployee Attrition Prediction (Held-Out Test Set) '
    f'| AUC-ROC: {aucRocTest:.3f} | F1: {f1Test:.3f}',
    fontsize=12
)

plt.tight_layout()

#saveTheConfusionMatrixHeatmapAsPNGSoItCanBeEmbeddedInTheAssessmentReportWithout
#RequiringTheReaderToReRunTheScript
confusionOutputFilePath = 'v3_haiku_confusion_matrix.png'
plt.savefig(confusionOutputFilePath, dpi=150)
plt.show()
plt.close()

print(f"Confusion matrix heatmap saved to: {confusionOutputFilePath}")

print("\n--- Final Summary ---")
print(f"Total flagged employees (predicted attrition > 0.5): {len(flaggedEmployees)}")
print(f"Flagged employees as percentage of test set: {len(flaggedEmployees) / len(yTest) * 100:.2f}%")
