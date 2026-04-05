import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load the dataset from the CSV file
rawData = pd.read_csv("IBM_HR_Analytics_Employee_Attrition_and_Performance.csv")

#display the first few rows to understand the structure of the dataset
print(rawData.head())
print(rawData.shape)
print(rawData.dtypes)

#create a copy of the dataset to avoid modifying the original data
employeeData = rawData.copy()

#initialise a LabelEncoder instance to encode categorical variables
labelEncoder = LabelEncoder()

#loop through each column and apply label encoding to categorical (object) columns
for columnName in employeeData.columns:
    if employeeData[columnName].dtype == 'object':
        employeeData[columnName] = labelEncoder.fit_transform(employeeData[columnName])

#separate the features (X) and the target variable (y)
#the target variable is 'Attrition' which we want to predict
featureColumns = [col for col in employeeData.columns if col != 'Attrition']
xFeatures = employeeData[featureColumns]
yTarget = employeeData['Attrition']

#split the data into training and testing sets using an 80/20 split
#random_state is set to 42 to ensure reproducibility
xTrain, xTest, yTrain, yTest = train_test_split(xFeatures, yTarget, test_size=0.2, random_state=42)

print(f"\nTraining set size: {xTrain.shape[0]} samples")
print(f"Testing set size: {xTest.shape[0]} samples")

#initialise the Logistic Regression classifier
#max_iter is increased to 1000 to ensure the model converges
logisticModel = LogisticRegression(max_iter=1000, random_state=42)

#train the logistic regression model on the training data
logisticModel.fit(xTrain, yTrain)

#use the trained model to make predictions on the test set
yPredicted = logisticModel.predict(xTest)

#calculate the accuracy score of the model on the test set
accuracyScore = accuracy_score(yTest, yPredicted)
print(f"\nModel Accuracy: {accuracyScore:.4f}")

#print a detailed classification report showing precision, recall, and F1-score
classificationReport = classification_report(yTest, yPredicted, target_names=['No Attrition', 'Attrition'])
print("\nClassification Report:")
print(classificationReport)

#compute the confusion matrix to evaluate the performance of the classifier
confusionMatrix = confusion_matrix(yTest, yPredicted)
print("\nConfusion Matrix:")
print(confusionMatrix)

#plot the confusion matrix as a heatmap using matplotlib
figureSize = (8, 6)
fig, ax = plt.subplots(figsize=figureSize)

#create the heatmap manually using imshow
heatmapImage = ax.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(heatmapImage, ax=ax)

#set axis labels and title
classLabels = ['No Attrition', 'Attrition']
tickMarks = np.arange(len(classLabels))
ax.set_xticks(tickMarks)
ax.set_yticks(tickMarks)
ax.set_xticklabels(classLabels)
ax.set_yticklabels(classLabels)

#add the count values as text inside each cell of the heatmap
thresholdValue = confusionMatrix.max() / 2.0
for rowIndex in range(confusionMatrix.shape[0]):
    for colIndex in range(confusionMatrix.shape[1]):
        cellValue = confusionMatrix[rowIndex, colIndex]
        textColour = 'white' if cellValue > thresholdValue else 'black'
        ax.text(colIndex, rowIndex, str(cellValue), ha='center', va='center', color=textColour, fontsize=14)

#add axis labels and a descriptive title to the plot
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix - Logistic Regression Employee Attrition', fontsize=13)

plt.tight_layout()

#save the confusion matrix heatmap as a PNG file
outputFilePath = '/mnt/user-data/outputs/confusion_matrix.png'
plt.savefig(outputFilePath, dpi=150)
plt.close()

print(f"\nConfusion matrix saved to: {outputFilePath}")