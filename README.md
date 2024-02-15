# credit-risk-classification
This repository is for the edX Data Visualization Bootcamp affiliated with Case Western Reserve University. The assignment in this repository is for Module 20 - supervised learning. In the file "credit_risk_classification_MH.ipynb" (found in the Credit_Risk folder) I used the skills I learned in this module, along with the provided starter code to create machine learning models. The data for these models came from the file "lending_data.csv" in the Resources folder (within the Credit_Risk folder).

The following is a Credit Risk Analysis Report based on the template provided in the starter code for this assignment.

## Overview of the Analysis

The purpose of analyzing the data is to reflect on what this code accomplished, understand why it was created, and discuss what was accomplished. The financial data that was used (from lending_data.csv) gave the size of the loan, the interest rate, the borrower income, the debt to income ratio, the number of accounts, any derogatory marks, the total debt, and the classification of the loan. The classification of the loan (loan_status) is what the model was created to predict based on the other data. After separating out the target category (loan_status), I used the value_counts function to see how many loans were in each category of loan. There were 75,036 loans in the healthy loan (0) category and 2,500 loans in the high risk (1) category.

I split the data into datasets using the train_test_split function from sklearn.model_selection. This function separated the input data (X) into a dataset for training the model (X_train) and a dataset to test the model's ability to predict (X_test). The function also separated the target (y) data into a dataset for training the model (y_train) and a dataset to evaluate the model's predictions against (y_test). I created a Logistic Regression Model using LogisticRegression from sklearn.linear_model and fit the instance of the Logistic Regression model with the training data (X_train, y_train). I then used the trained model to predict what category a loan would fall into (y) based on the X_test dataset. I calculated the accuracy score of the model, generated a confusion matrix, and created a classification report based on the comparison between the predictions made with the model on the X_test dataset and the y_test dataset which were the actual categorizations of the loans with the X_test inputs. After this part of the analysis, I resampled the training data using RandomOverSampler from imblearn.over_sampling. I then created a logistic regression model to be trained with the resampled data and used it to predict y output with the X_test data. I did the same types of analysis on the predictions versus the acutal output that I did using the first instance of the logistic regression model.

## Results

Below are the balanced accuracy scores, the precision scores, and the recall scores for both models I created. The accuracy score describes how often the model is correct - the ratio of correctly predicted observations to the total number of observations. The precision score represents the ratio of correctly predicted positive observations to the total predicted positive observations; this describes how many of the loans that are classified in a category will actually be in that category - a high precision score means little chance for false positive results. The recall score represents the ratio of correctly predicted positive observations to all predicted observations for that class; this describes how many loans are correctly classified out of all the loans in that category - a high recall score means little chance for false negative results.

The categories (y) for classifying a loan based on input data (X) are 0 - healthy loan and 1 - high-risk loan.

* Machine Learning Model 1:
  * Model 1 was the linear regression model based on the original dataset.
  
  Balanced Accuracy Score: 0.952
  
  This score suggests that the loan classifications made with the model are correct about 95 out of 100 loans.

  Precision: 0 = 1.00, 1 = 0.85
  
  This score suggests that when a loan is in the 0 (healthy loan) category, the number of loans in that category is almost exactly equal to the true number of loans that should be in that category. The score for the 1 (high-risk loan) category suggests that only 85 out of 100 loans in the 1 category should actually be in that category.

  Recall: 0 = 0.99, 1 = 0.85
  
  The score for the 0 (healthy loan) category suggests that 1 out of 100 loans in the 0 category is not meant to be in that category. THe score for the 1 (high-risk loan) category suggests that 15 out of 100 loans in the 1 category are not meant to be in that category.



* Machine Learning Model 2:
  * Model 2 was the linear regression model using resampled training data.

Balanced Accuracy Score: 0.994

This score suggests that the loan classifications made with the model are correct about 99 out of 100 loans.

Precision: 0 = 1.00, 1 = 0.84

This score suggests that when a loan is in the 0 (healthy loan) category, the number of loans in that category is alomst exactly equal to the true number of loans that should be in that category. The score for the 1 (high-risk loan) category suggests that only 84 out of 100 loans in the 1 category should actually be in that category.

Recall: 0 = 0.99, 1 = 0.99

The score for the 0 (healthy loan) category suggests that 1 out of 100 loans in the 0 category is not meant to be in that category. The score for the 1 (high-risk loan) category suggests that 1 out of 100 loans in the 1 category is not meant to be in that category.

## Summary

Looking at the results of the analysis of the output of both machine learning models, the second model (with the resampled data) appears to perform better than the first model. This observation is based on the higher accuracy and recall scores seen in the second model when compared to the first model. Both the first and second models had similar precision scores of 1.00 for the 0 (healthy loan) category and 0.85/0.84 for the 1 (high-risk loan) category. However, performance depends on the problem the model is trying to solve. These models were created to evaluate which loans should be put in the high risk category. Both models misclassify about 15% of the loans in the high-risk category - these loans should instead be considered healthy loans. Depending on the needs of the company that would use this tool to evaluate loans and the needs of the borrowers applying for loans, this amount of misclassification may be unacceptable. In the module that prepared me to complete this assignment, I learned that 85% accuracy is often the standard for real-world machine learning models. If that is the case, both models would be acceptable for evaluating loan risk, however, the second model reduces the chance for false negative results as compared to the first model so the second model is the one that I recommend.
