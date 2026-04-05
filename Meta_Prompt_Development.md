# Meta Prompt Development

This file documents the iterative process used to engineer the final code generation prompt (1f) used to produce V1a and V1b. Rather than prompting an LLM directly, a meta prompt asks the LLM to write the prompt for the user, yielding more precise and consistent results

---

## 1a — Claude's Initial Meta Prompt (Engineered to Fail)
**Tool:** Claude Sonnet 4.6  
**Purpose:** Deliberately constructed to produce a weak baseline by instructing the model to ignore class imbalance, use accuracy only, and include protected characteristics.

> You are helping me generate an initial piece of Python code for a university machine learning coursework. The task is binary classification: predicting employee attrition (Yes/No) using the IBM HR Analytics Employee Attrition dataset from Kaggle (1,470 rows, 35 features).
Generate a prompt I can use to produce beginner-level Python code that a student new to machine learning would write. The code should:
•	Use logistic regression as the classifier
•	Use LabelEncoder for all categorical variables
•	Include Age, Gender and MaritalStatus as training features
•	Use accuracy as the only evaluation metric
•	Use a single train/test split with no cross-validation
•	Use sklearn, pandas and matplotlib only
The code should be functional and produce a result, but should not address class imbalance, should not question whether accuracy is appropriate, and should not raise any concerns about protected attributes being used as features.
Return only the refined prompt, nothing else.

---

## 1b — Claude's Revised Meta Prompt (Partial Fix, Subtle Failures Remain)
**Tool:** Claude Sonnet 4.6  
**Purpose:** Removed the most explicit failure instructions but subtle ones remained — LabelEncoder still applied to nominal variables, no cross-validation, no class weighting.

> You are helping me generate an initial piece of Python code for a university machine learning coursework. The task is binary classification: predicting employee attrition (Yes/No) using the IBM HR Analytics Employee Attrition dataset from Kaggle (1,470 rows, 35 features).
Generate a prompt I can use to produce beginner-level Python code that a student new to machine learning would write. The code should:
•	Import pandas, numpy, matplotlib and sklearn
•	Use logistic regression as the classifier
•	Use LabelEncoder for all categorical variables including nominal ones such as JobRole, Department and MaritalStatus
•	Include Age, Gender and MaritalStatus as training features without questioning whether this is appropriate
•	Use a single train/test split with no cross validation
•	Use accuracy as the only evaluation metric without questioning whether it is appropriate for imbalanced data
•	Use sklearn, pandas, numpy and matplotlib only
The code should be functional and produce a result, but should not address class imbalance, should not question whether accuracy is appropriate on skewed data, and should not raise any concerns about protected attributes being included as features.
Return only the refined prompt, nothing else.

---

## 1c — Claude's Final Meta Prompt (All Explicit Failures Removed)
**Tool:** Claude Sonnet 4.6  
**Purpose:** Removed all deliberate failure instructions to produce a neutral, clean meta prompt with no engineered weaknesses.

> You are helping me generate a Python code generation prompt for a university machine learning coursework. The task is binary classification: predicting employee attrition (Yes/No) using an HR dataset I will provide.
Generate a prompt that asks for the following:
•	Import and use pandas, numpy, matplotlib and sklearn
•	Use logistic regression as the classifier
•	Use LabelEncoder for all categorical variables
•	Use all available features
•	Use a single train/test split
•	Include an evaluation metric
Return only the refined prompt, nothing else.

---

## 1d — ChatGPT's Output from 1c (GPT-5)
**Tool:** GPT-5  
**Purpose:** GPT-5's interpretation of 1c — produced a simpler meta prompt by removing all library-specific mentions and coding preferences.

> Write a complete Python script to perform a binary classification task predicting employee attrition (Yes/No) using a provided HR dataset. The script must:
•	Import and use pandas, numpy, matplotlib, and sklearn 
•	Load the dataset and preprocess it appropriately 
•	Encode all categorical variables using LabelEncoder 
•	Use all available features for prediction 
•	Split the data into training and testing sets using a single train/test split 
•	Train a Logistic Regression model 
•	Evaluate the model using at least one appropriate metric (e.g., accuracy, confusion matrix) 
•	Include clear comments explaining each step of the process
---

## 1e — Claude's Output from 1c (Sonnet 4.6)
**Tool:** Claude Sonnet 4.6  
**Purpose:** Claude's interpretation of 1c — kept every detail unchanged, preserving all library-specific mentions and coding conventions.

> Write a Python script that performs binary classification to predict employee attrition (Yes/No) using the IBM HR Analytics Employee Attrition dataset.
The script must:
•	Import pandas, numpy, matplotlib, and sklearn
•	Load the dataset from a CSV file
•	Use LabelEncoder from sklearn.preprocessing to encode all categorical variables
•	Use all available features in the dataset
•	Split the data into training and testing sets using a single train/test split with sklearn.model_selection.train_test_split
•	Train a Logistic Regression classifier from sklearn.linear_model on the training set
•	Evaluate the model on the test set using accuracy score, a classification report, and a confusion matrix
•	Plot the confusion matrix as a heatmap using matplotlib and save it as a PNG file
Follow these coding conventions strictly:
•	All variable names must use camelCase where the first word is fully lowercase and each subsequent word starts with an uppercase letter (e.g. trainData, labelEncoder, confusionMatrix)
•	All comments must begin with #comment (no space between the hash and the word comment), for example: #this is a comment
Return only the complete Python script with no explanation.

---

## 1f — Final Code Generation Prompt (Best of 1d and 1e)
**Tool:** Run on GPT-5 to produce V1a and on Claude Sonnet 4.6 to produce V1b  
**Purpose:** Combined the clarity of 1d with the technical precision of 1e to produce the final balanced prompt used to generate both baseline versions.

>  Write a complete Python script that performs binary classification to predict employee attrition (Yes/No) using the IBM HR Analytics Employee Attrition dataset.
The script must:
•	Import pandas, numpy, matplotlib and sklearn
•	Load the dataset from a CSV file
•	Use LabelEncoder from sklearn.preprocessing to encode all categorical variables
•	Use all available features in the dataset
•	Split the data into training and testing sets using a single train/test split with sklearn.model_selection.train_test_split
•	Train a Logistic Regression classifier from sklearn.linear_model on the training set
•	Evaluate the model on the test set using accuracy score, a classification report and a confusion matrix
•	Plot the confusion matrix as a heatmap using matplotlib and save it as a PNG file
•	Include clear comments in plain English explaining each step of the process
Follow these coding conventions strictly:
•	All variable names must use camelCase where the first word is fully lowercase and each subsequent word starts with an uppercase letter (e.g. trainData, labelEncoder, confusionMatrix)
•	All comments must begin with #comment with no space between the hash and the text (e.g. #load the dataset)
Return only the complete Python script with no explanation.
