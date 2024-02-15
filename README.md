# credit-risk-classification
This repository is for the edX Data Visualization Bootcamp affiliated with Case Western Reserve University. The assignment in this repository is for Module 20 - supervised learning. In the file "credit_risk_classification_MH.ipynb" (found in the Credit_Risk folder) I used the skills I learned in this module, along with the provided starter code to create machine learning models. The data for these models came from the file "lending_data.csv" in the Resources folder (within the Credit_Risk folder).

The following is a Credit Risk Analysis Report based on the template provided in the starter code for this assignment.

## Overview of the Analysis

The purpose of analyzing the data is to reflect on what this code accomplished, understand why it was created, and discuss what was accomplished. The financial data that was used (from lending_data.csv) gave the size of the loan, the interest rate, the borrower income, the debt to income ratio, the number of accounts, any derogatory marks, the total debt, and the classification of the loan. The classification of the loan (loan_status) is what the model was created to predict based on the other data. After separating out the target category (loan_status), I used the value_counts function to see how many loans were in each category of loan. There were 75,036 loans in the healthy loan (0) category and 2,500 loans in the high risk (1) category.

I split the data into datasets using the train_test_split function from sklearn.model_selection. This function separated the input data (X) into a dataset for training the model (X_train) and a dataset to test the model's ability to predict (X_test). The function also separated the target (y) data into a dataset for training the model (y_train) and a dataset to evaluate the model's predictions against (y_test). I created a Logistic Regression Model using LogisticRegression from sklearn.linear_model and fit the instance of the Logistic Regression model with the training data (X_train, y_train). I then used the trained model to predict what category a loan would fall into (y) based on the X_test dataset. I calculated the accuracy score of the model, generated a confusion matrix, and created a classification report based on the comparison between the predictions made with the model on the X_test dataset and the y_test dataset which were the actual categorizations of the loans with the X_test inputs. After this part of the analysis, I resampled the training data using RandomOverSampler from imblearn.over_sampling. I then created a logistic regression model to be trained with the resampled data and used it to predict y output with the X_test data. I did the same types of analysis on the predictions versus the acutal output that I did using the first instance of the logistic regression model.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  Balanced Accuracy Score: 0.952
  Precision: 0 = 1.00, 1 = 0.85
  Recall: 0 = 0.99, 1 = 0.85



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
Balanced Accuracy Score: 0.994
Precision: 0 = 1.00, 1 = 0.84
Recall: 0 = 0.99, 1 = 0.99

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

