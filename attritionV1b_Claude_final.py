import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load the dataset from the CSV file
rawData = pd.read_csv('IBM_HR_Analytics_Employee_Attrition_and_Performance.csv')

#display basic information about the dataset to understand its structure
print("Dataset shape:", rawData.shape)
print("\nFirst five rows:")
print(rawData.head())
print("\nColumn data types:")
print(rawData.dtypes)

#create a copy of the dataset to avoid modifying the original data
employeeData = rawData.copy()

#initialise a LabelEncoder instance to encode categorical variables into numeric values
labelEncoder = LabelEncoder()

#loop through each column and apply label encoding to all string or categorical columns
#using is_string_dtype to catch both legacy object dtype and newer pandas StringDtype
for columnName in employeeData.columns:
    if pd.api.types.is_string_dtype(employeeData[columnName]):
        employeeData[columnName] = labelEncoder.fit_transform(
            employeeData[columnName].astype(str)
        )

#confirm that all columns have been successfully converted to numeric types
print("\nAll columns converted to numeric. Remaining non-numeric columns:",
      employeeData.select_dtypes(exclude=[np.number]).columns.tolist())

#separate the feature matrix (xFeatures) from the target variable (yTarget)
#the target column 'Attrition' contains 0 for No and 1 for Yes after encoding
featureColumns = [col for col in employeeData.columns if col != 'Attrition']
xFeatures = employeeData[featureColumns]
yTarget = employeeData['Attrition']

#split the data into training and testing sets using an 80/20 ratio
#random_state is fixed at 42 to ensure the same split each time the script is run
xTrain, xTest, yTrain, yTest = train_test_split(
    xFeatures, yTarget, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {xTrain.shape[0]} samples")
print(f"Testing set size:  {xTest.shape[0]} samples")

#initialise the Logistic Regression classifier
#max_iter is set to 5000 to allow the solver enough iterations to converge
#solver is set to saga which handles larger datasets efficiently
logisticModel = LogisticRegression(max_iter=5000, solver='saga', random_state=42)

#train the model using the training features and labels
logisticModel.fit(xTrain, yTrain)

#generate predictions on the unseen test set
yPredicted = logisticModel.predict(xTest)

#calculate overall accuracy as the proportion of correct predictions
accuracyScore = accuracy_score(yTest, yPredicted)
print(f"\nModel Accuracy: {accuracyScore:.4f} ({accuracyScore * 100:.2f}%)")

#print a full classification report showing precision recall and F1-score per class
classificationReport = classification_report(
    yTest, yPredicted, target_names=['No Attrition', 'Attrition']
)
print("\nClassification Report:")
print(classificationReport)

#compute the confusion matrix to show true positives false positives and so on
confusionMatrix = confusion_matrix(yTest, yPredicted)
print("Confusion Matrix:")
print(confusionMatrix)

#plot the confusion matrix as a colour-coded heatmap using matplotlib
figurePlot, axisPlot = plt.subplots(figsize=(8, 6))

#render the confusion matrix values as a blue gradient heatmap
heatmapImage = axisPlot.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)

#add a colour bar on the right-hand side to indicate the scale
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
    'Confusion Matrix - Logistic Regression\nEmployee Attrition Prediction', fontsize=13
)

plt.tight_layout()

#save the finished heatmap as a PNG file in the current working directory
outputFilePath = 'v1b_Claude_final_confusion_matrix.png'
plt.savefig(outputFilePath, dpi=150)
plt.show()
plt.close()

print(f"\nConfusion matrix heatmap saved to: {outputFilePath}")