#comment import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#comment load the dataset from csv file
dataFrame = pd.read_csv("IBM_HR_Analytics_Employee_Attrition_and_Performance.csv")

#comment fill any missing values to prevent errors
dataFrame = dataFrame.fillna(0)

#comment encode all categorical columns safely
labelEncoder = LabelEncoder()

for column in dataFrame.columns:
    if dataFrame[column].dtype == "object":
        dataFrame[column] = dataFrame[column].astype(str)
        dataFrame[column] = labelEncoder.fit_transform(dataFrame[column])

#comment verify all columns are numeric
print(dataFrame.dtypes)

#comment define features and target variable
features = dataFrame.drop("Attrition", axis=1)
target = dataFrame["Attrition"]

#comment split the data into training and testing sets
trainData, testData, trainLabels, testLabels = train_test_split(
    features, target, test_size=0.2, random_state=42
)

#comment initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(trainData, trainLabels)

#comment make predictions on test data
predictions = model.predict(testData)

#comment evaluate the model
accuracy = accuracy_score(testLabels, predictions)
report = classification_report(testLabels, predictions)
confusionMatrix = confusion_matrix(testLabels, predictions)

#comment print evaluation results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", confusionMatrix)

#comment plot confusion matrix as heatmap using matplotlib
plt.figure()
plt.imshow(confusionMatrix)
plt.title("Confusion Matrix Heatmap")
plt.colorbar()

#comment add axis labels
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

#comment add tick marks
tickMarks = np.arange(len(np.unique(target)))
plt.xticks(tickMarks)
plt.yticks(tickMarks)

#comment annotate each cell with values
for i in range(confusionMatrix.shape[0]):
    for j in range(confusionMatrix.shape[1]):
        plt.text(j, i, confusionMatrix[i, j], ha="center", va="center")

#comment save the figure as png file
plt.savefig("v1a_ChatGPT_fix1_confusionMatrix.png")

#comment display the plot
plt.show()