import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, auc, roc_curve
import warnings
warnings.filterwarnings('ignore')

#LoadDatasetFromCSV
rawData = pd.read_csv('IBM_HR_Analytics_Employee_Attrition_and_Performance.csv')

print("Dataset shape:", rawData.shape)
print("\nFirst five rows:")
print(rawData.head())
print("\nColumn data types:")
print(rawData.dtypes)
print("\nTarget variable distribution:")
print(rawData['Attrition'].value_counts())
print("Percentage distribution:")
print(rawData['Attrition'].value_counts(normalize=True) * 100)

#CreateWorkingCopy
employeeData = rawData.copy()

#DropConstantColumnsAsTheyProvideNoDiscriminativePower
columnsToDropConstant = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
employeeData = employeeData.drop(columns=columnsToDropConstant)
print(f"\nDropped constant columns: {columnsToDropConstant}")

#EncodeTargetVariableBeforeAuditExtraction
employeeData['Attrition'] = (employeeData['Attrition'] == 'Yes').astype(int)

#ExtractProtectedAttributeAuditDataframeBeforeTrainingToEnsureNoLeakage
auditData = employeeData[['Age', 'Gender', 'MaritalStatus']].copy()
print("\nAudit dataframe shape:", auditData.shape)
print("Audit dataframe sample:")
print(auditData.head())

#RemoveProtectedAttributesFromFeatureMatrix
employeeData = employeeData.drop(columns=['Age', 'Gender', 'MaritalStatus'])

#IdentifyFeatureTypesForProperEncoding
nominalCategoricalColumns = ['JobRole', 'Department', 'EducationField', 'BusinessTravel']
binaryOrdinalColumns = ['OverTime']
#EducationLevelIsOrdinalSoKeepAsNumeric
continuousNumericColumns = [col for col in employeeData.columns 
                           if col not in nominalCategoricalColumns + binaryOrdinalColumns + ['Attrition']]

print(f"\nNominal categorical features: {nominalCategoricalColumns}")
print(f"Binary ordinal features: {binaryOrdinalColumns}")
print(f"Continuous numeric features: {continuousNumericColumns}")

#SeparateTargetFromFeatures
xFeatures = employeeData.drop(columns=['Attrition'])
yTarget = employeeData['Attrition']

#EncodeOverTimeAsBinaryBecauseItHasOnlyTwoCategories
binaryEncoder = {'Yes': 1, 'No': 0}
xFeatures['OverTime'] = xFeatures['OverTime'].map(binaryEncoder)

#CreatePreprocessingPipelineWithOneHotEncoderAndStandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        #OneHotEncoderForNominalCategoricalVariablesAvoidsImposingFalseOrdinalRelationships
        ('onehotCategories', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         nominalCategoricalColumns),
        #StandardScalerForContinuousNumericFeaturesToEnsureEqualWeightingAcrossScales
        ('scaleNumeric', StandardScaler(), continuousNumericColumns)
    ],
    remainder='passthrough'
)

#CreateFullPipelineWithPreprocessingAndLogisticRegression
fullPipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #LogisticRegressionWithClassWeightBalancedAddressesClassImbalanceBy16PercentAttritionRate
    ('classifier', LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', random_state=42))
])

print("\n" + "="*70)
print("STRATIFIED 5-FOLD CROSS-VALIDATION WITH BALANCED CLASS WEIGHT")
print("="*70)

#StratifiedKFoldPreservesClassDistributionAcrossEveryFoldToAvoidBiasedEstimates
stratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#DefineMetricsToEvaluatePositiveClassPerformanceNotOverallAccuracy
scoringMetrics = {
    'roc_auc': 'roc_auc',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall'
}

#PerformCrossValidationAndCollectMetrics
crossValResults = cross_validate(fullPipeline, xFeatures, yTarget, cv=stratifiedKFold, 
                                  scoring=scoringMetrics, return_train_score=False)

#PrintCrossValidationResults
print("\nCross-Validation Results (5 Folds):")
print(f"AUC-ROC: {crossValResults['test_roc_auc'].mean():.4f} (+/- {crossValResults['test_roc_auc'].std():.4f})")
print(f"F1-Score: {crossValResults['test_f1'].mean():.4f} (+/- {crossValResults['test_f1'].std():.4f})")
print(f"Precision: {crossValResults['test_precision'].mean():.4f} (+/- {crossValResults['test_precision'].std():.4f})")
print(f"Recall: {crossValResults['test_recall'].mean():.4f} (+/- {crossValResults['test_recall'].std():.4f})")

print("\nPer-Fold Breakdown:")
for fold in range(5):
    print(f"Fold {fold+1}: AUC={crossValResults['test_roc_auc'][fold]:.4f}, "
          f"F1={crossValResults['test_f1'][fold]:.4f}, "
          f"Precision={crossValResults['test_precision'][fold]:.4f}, "
          f"Recall={crossValResults['test_recall'][fold]:.4f}")

#TrainFinalModelOnEntireDatasetForConfusionMatrixGeneration
fullPipeline.fit(xFeatures, yTarget)

#GeneratePredictionsAndProbabilitiesForFinalEvaluation
yPredicted = fullPipeline.predict(xFeatures)
yPredictedProba = fullPipeline.predict_proba(xFeatures)[:, 1]

#ComputeFinalMetricsOnFullDatasetForCompletePictureOfModelPerformance
finalAucScore = roc_auc_score(yTarget, yPredictedProba)
finalF1Score = f1_score(yTarget, yPredicted)
finalPrecisionScore = precision_score(yTarget, yPredicted)
finalRecallScore = recall_score(yTarget, yPredicted)

print("\n" + "="*70)
print("FINAL MODEL EVALUATION ON FULL DATASET")
print("="*70)
print(f"AUC-ROC Score: {finalAucScore:.4f}")
print(f"F1-Score: {finalF1Score:.4f}")
print(f"Precision: {finalPrecisionScore:.4f}")
print(f"Recall: {finalRecallScore:.4f}")

#ComputeConfusionMatrixToIdentifyTruePositivesFalsePositivesAndFalseNegatives
confusionMatrixFinal = confusion_matrix(yTarget, yPredicted)
print("\nConfusion Matrix:")
print(confusionMatrixFinal)

#ExtractConfusionMatrixComponentsForInterpretation
trueNegative = confusionMatrixFinal[0, 0]
falsePositive = confusionMatrixFinal[0, 1]
falseNegative = confusionMatrixFinal[1, 0]
truePositive = confusionMatrixFinal[1, 1]

print(f"\nTrue Negatives (Correctly predicted non-attrition): {trueNegative}")
print(f"False Positives (Incorrectly predicted attrition): {falsePositive}")
print(f"False Negatives (Missed attrition cases): {falseNegative}")
print(f"True Positives (Correctly predicted attrition): {truePositive}")

#PlotConfusionMatrixAsHeatmapForVisualInterpretation
figurePlot, axisPlot = plt.subplots(figsize=(8, 6))

#RenderConfusionMatrixAsColorCodedHeatmapWithBlueGradient
heatmapImage = axisPlot.imshow(confusionMatrixFinal, interpolation='nearest', cmap=plt.cm.Blues)

#AddColorBarToIndicateValueScale
plt.colorbar(heatmapImage, ax=axisPlot)

#DefineClassLabelsAndSetTickPositions
classLabels = ['No Attrition', 'Attrition']
tickPositions = np.arange(len(classLabels))
axisPlot.set_xticks(tickPositions)
axisPlot.set_yticks(tickPositions)
axisPlot.set_xticklabels(classLabels, fontsize=12)
axisPlot.set_yticklabels(classLabels, fontsize=12)

#AddNumericCountInsideEachCellWithContrastingTextColorForReadability
thresholdValue = confusionMatrixFinal.max() / 2.0
for rowIndex in range(confusionMatrixFinal.shape[0]):
    for colIndex in range(confusionMatrixFinal.shape[1]):
        cellValue = confusionMatrixFinal[rowIndex, colIndex]
        textColor = 'white' if cellValue > thresholdValue else 'black'
        axisPlot.text(
            colIndex, rowIndex, str(cellValue),
            ha='center', va='center', color=textColor, fontsize=14, fontweight='bold'
        )

#LabelAxesAndAddDescriptiveTitle
axisPlot.set_xlabel('Predicted Label', fontsize=13)
axisPlot.set_ylabel('True Label', fontsize=13)
axisPlot.set_title(
    'Confusion Matrix - Logistic Regression with Class Weight Balancing\nEmployee Attrition Prediction', 
    fontsize=13
)

plt.tight_layout()

#SaveConfusionMatrixHeatmapAsPNGFileInCurrentWorkingDirectory
outputFilePath = '/mnt/user-data/outputs/v2_improved_confusion_matrix.png'
plt.savefig(outputFilePath, dpi=150)
plt.show()
plt.close()

print(f"\nConfusion matrix heatmap saved to: {outputFilePath}")

#VerifyProtectedAttributesNotInFeatureMatrix
print("\n" + "="*70)
print("VERIFICATION: PROTECTED ATTRIBUTES NOT IN FEATURE MATRIX")
print("="*70)
print(f"Features shape: {xFeatures.shape}")
print(f"Features in training: {xFeatures.columns.tolist()[:10]}... (first 10)")
print(f"Age in features: {'Age' in xFeatures.columns}")
print(f"Gender in features: {'Gender' in xFeatures.columns}")
print(f"MaritalStatus in features: {'MaritalStatus' in xFeatures.columns}")
print(f"\nAudit dataframe preserved: Shape {auditData.shape}, Columns {auditData.columns.tolist()}")
