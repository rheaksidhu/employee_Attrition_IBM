import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    make_scorer, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline

#load the dataset from the CSV file
rawData = pd.read_csv('IBM_HR_Analytics_Employee_Attrition_and_Performance.csv')

#display basic information to confirm successful loading
print("Dataset shape:", rawData.shape)
print("\nFirst five rows:")
print(rawData.head())
print("\nColumn data types:")
print(rawData.dtypes)

#show class distribution to confirm the imbalance that motivates balanced class weights
print("\nTarget variable distribution:")
print(rawData['Attrition'].value_counts())

#create a working copy so the original data remains untouched for audit purposes
employeeData = rawData.copy()

#drop constant or identifier columns because they carry zero predictive variance
constantColumns = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
employeeData = employeeData.drop(columns=constantColumns)
print(f"\nDropped constant columns: {constantColumns}")

#encode the binary target variable before any feature processing
#mapping Yes to 1 and No to 0 so the positive class aligns with attrition
employeeData['Attrition'] = employeeData['Attrition'].map({'Yes': 1, 'No': 0})

#separate protected attributes into an audit dataframe to prevent the model
#from learning discriminatory patterns based on age gender or marital status
protectedColumns = ['Age', 'Gender', 'MaritalStatus']
auditData = employeeData[protectedColumns].copy()
employeeData = employeeData.drop(columns=protectedColumns)
print(f"Protected attributes removed from features and stored in audit dataframe: {protectedColumns}")

#separate the target variable from the feature matrix
yTarget = employeeData['Attrition']
xFeatures = employeeData.drop(columns=['Attrition'])

#verify that no protected attributes remain in the feature matrix
for colName in protectedColumns:
    assert colName not in xFeatures.columns, f"Protected attribute {colName} still in features"
print("\nConfirmed: no protected attributes in the feature matrix")

#identify nominal categorical columns that require one-hot encoding
#these have no inherent ordering so LabelEncoder would impose false ordinal relationships
nominalColumns = ['JobRole', 'Department', 'EducationField', 'BusinessTravel', 'OverTime']

#identify continuous numeric columns that benefit from standardisation
#scaling prevents features with large ranges from dominating the logistic regression coefficients
numericColumns = [col for col in xFeatures.columns if col not in nominalColumns]

print(f"\nNominal columns for OneHotEncoding: {nominalColumns}")
print(f"Numeric columns for StandardScaler: {numericColumns}")

#build a ColumnTransformer to apply the correct preprocessing to each column type in one step
#OneHotEncoder avoids false ordinal relationships in nominal categories
#StandardScaler centres and scales numeric features so the regularised model treats them equally
preprocessor = ColumnTransformer(
    transformers=[
        ('nominal', OneHotEncoder(drop='first', handle_unknown='error'), nominalColumns),
        ('numeric', StandardScaler(), numericColumns)
    ],
    remainder='drop'
)

#wrap preprocessing and classification into a single pipeline to prevent data leakage
#fitting the scaler and encoder inside cross-validation ensures no test information contaminates training
classifierPipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=5000,
        solver='saga',
        class_weight='balanced',
        random_state=42
    ))
])

#use StratifiedKFold to preserve the class imbalance ratio in every fold
#this prevents folds where the minority class is underrepresented from skewing evaluation
stratifiedFolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#define scorers that focus on the positive class because attrition is the event of interest
#accuracy is removed as a headline metric because it masks poor minority-class performance
scoringMetrics = {
    'f1': make_scorer(f1_score, pos_label=1),
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1),
    'roc_auc': 'roc_auc'
}

#run stratified cross-validation to get robust performance estimates across all folds
cvResults = cross_validate(
    classifierPipeline, xFeatures, yTarget,
    cv=stratifiedFolds, scoring=scoringMetrics, return_train_score=False
)

#report mean and standard deviation for each metric across all five folds
print("\n--- Stratified 5-Fold Cross-Validation Results ---")
print(f"AUC-ROC:   {cvResults['test_roc_auc'].mean():.4f} (+/- {cvResults['test_roc_auc'].std():.4f})")
print(f"F1-Score:  {cvResults['test_f1'].mean():.4f} (+/- {cvResults['test_f1'].std():.4f})")
print(f"Precision: {cvResults['test_precision'].mean():.4f} (+/- {cvResults['test_precision'].std():.4f})")
print(f"Recall:    {cvResults['test_recall'].mean():.4f} (+/- {cvResults['test_recall'].std():.4f})")

#fit the pipeline on the full dataset to generate a single confusion matrix for reporting
#this avoids choosing an arbitrary single fold and uses all available data for the final evaluation
classifierPipeline.fit(xFeatures, yTarget)
yPredicted = classifierPipeline.predict(xFeatures)

#generate the classification report for completeness showing both classes
classReport = classification_report(
    yTarget, yPredicted, target_names=['No Attrition', 'Attrition']
)
print("\nClassification Report (full-data refit):")
print(classReport)

#compute the confusion matrix for the heatmap visualisation
confusionMatrix = confusion_matrix(yTarget, yPredicted)
print("Confusion Matrix:")
print(confusionMatrix)

#plot the confusion matrix as a colour-coded heatmap for intuitive interpretation
figurePlot, axisPlot = plt.subplots(figsize=(8, 6))

#render the confusion matrix values as a blue gradient heatmap
heatmapImage = axisPlot.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)

#add a colour bar on the right-hand side to indicate the scale of counts
plt.colorbar(heatmapImage, ax=axisPlot)

#define the class labels and set tick positions on both axes
classLabels = ['No Attrition', 'Attrition']
tickPositions = np.arange(len(classLabels))
axisPlot.set_xticks(tickPositions)
axisPlot.set_yticks(tickPositions)
axisPlot.set_xticklabels(classLabels, fontsize=12)
axisPlot.set_yticklabels(classLabels, fontsize=12)

#add the numeric count inside each cell with contrasting text colour for readability
thresholdValue = confusionMatrix.max() / 2.0
for rowIndex in range(confusionMatrix.shape[0]):
    for colIndex in range(confusionMatrix.shape[1]):
        cellValue = confusionMatrix[rowIndex, colIndex]
        textColour = 'white' if cellValue > thresholdValue else 'black'
        axisPlot.text(
            colIndex, rowIndex, str(cellValue),
            ha='center', va='center', color=textColour, fontsize=14, fontweight='bold'
        )

#label the axes and add a descriptive title to the heatmap
axisPlot.set_xlabel('Predicted Label', fontsize=13)
axisPlot.set_ylabel('True Label', fontsize=13)
axisPlot.set_title(
    'Confusion Matrix - Logistic Regression (Balanced)\nEmployee Attrition Prediction', fontsize=13
)

plt.tight_layout()

#save the finished heatmap as a PNG file in the current working directory
outputFilePath = 'v2c_Opus_confusion_matrix.png'
plt.savefig(outputFilePath, dpi=150)
plt.close()

print(f"\nConfusion matrix heatmap saved to: {outputFilePath}")
