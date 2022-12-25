# Customer Churn Prediction

<img src="/images/subscribe.jpg" width="400" align = "center">
#### -- Project Status: [Completed]

## Project Motivations:
Customer churn can have a significant impact on a company's bottom line. Losing customers means losing the revenue they generate, and acquiring new customers can be a costly and time-consuming process. By predicting which customers are at risk of churning, a company can take proactive steps to retain them and minimize the financial impact of churn.

In this project, we aim to develop a machine learning model that can accurately predict customer churn. This model can be used by the company to identify at-risk customers and take targeted retention efforts, ultimately resulting in improved customer retention and profitability.

Additionally, understanding the factors that contribute to customer churn can help a company identify areas for improvement in its products, services, or customer experience. By analyzing the characteristics of churned customers and comparing them to those who have not churned, the company can identify potential areas for intervention and make changes to reduce churn in the future.

Overall, this customer churn prediction project has the potential to provide valuable insights and drive business impact for the company.


This data science project aims to predict customer churn for a fictional company using machine learning techniques. 
Churn, also known as customer attrition, refers to the loss of customers over a certain period of time. 
Predicting churn can help a company identify at-risk customers and take steps to retain them.

## Project Description

This project aims to predict customer churn for a fictional company. 
Churn, also known as customer attrition, refers to the loss of customers over a certain period of time. 
Predicting churn can help a company identify at-risk customers and take steps to retain them.

### About the dataset

The data used in this project is a fictional telecom dataset containing customer information whether or not they have churned. 
The data is provided in the data folder and is in a CSV format. For more infomration on the dataset see [here](https://www.kaggle.com/c/customer-churn-prediction-2020)

### Methods Used
* Data Cleaning and Wrangling
* Data Analysis
* Data Visualization
* Data Augmentation
* Machine Learning Model: Logistic Regression, Random Forest, XGBoost
* Hyperparameter tuning

### Prerequisites 

To run this project, you will need:

Python 3.6 or higher
Jupyter Notebook
pandas
scikit-learn
xgboost

### Installing

**Running the analysis**

To run the analysis, follow the steps in the Jupyter Notebook [See here](https://github.com/avtnguyen/customer-churn-predictions/blob/main/customer_churn_model.ipynb). The notebook includes data exploration, preprocessing, and model training and evaluation.

**Built With**

* Python - Programming language

* Jupyter Notebook - Interactive coding environment

* pandas - Data manipulation and analysis library

* scikit-learn - Machine learning library

* xgboost - Gradient boosting algorithm

* matplotlib and searborn - Data visualization

### Project pipeline
1. Data acquisition and exploratory data anslysis (EDA):
The first step in the project is to obtain and prepare the data for analysis. 
This involves collecting data from multiple sources, cleaning and formatting the data, and performing any necessary preprocessing steps.
```
There are 19 features: ['state', 'account_length', 'area_code', 'international_plan', 'voice_mail_plan', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls', 'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls']

The label is churn that takes two values no: customer not leaving, and yes: customer leaving with the following summary
Train dataset:
 no     3652
yes     598
Name: churn, dtype: int64
```
Next, we will perform EDA to understand the characteristics of the data and identify any patterns or trends that may be relevant to the churn prediction task. 
This will involve visualizing the data, calculating summary statistics, and identifying correlations or other relationships between variables. (See exemple below)
<img src="/images/hisplot.png" width="500" align = "center">

2. Feature engineering
Based on the insights gained from the EDA, we will then select and create relevant features to use as inputs for the machine learning model. 
Specifically, we will transform categorical variables to integer values based on the following scheme:
* Label encoding for the following features: international_plan, voice_mail_plan, churn, area_code
* One-hot encoding for the following features: states. This will help avoid the problems of ordinal data and biases in the model. 
Here, we don't have a lot of features and thus, it won't lead to computational and storage overhead

Furthermore, since the churn dataset is an imbalanced, I also test several data augmentation techniques including SMOTE and SMOTE+TOMEK. 
SMOTE is a technique that generates synthetic data points for the minority class by interpolating between existing minority class data points. 
The synthetic data points are generated in such a way that they are similar to the original data points, but are not exact copies. 
This helps to balance the class distribution and improve the model's performance on the minority class.

In addition, SMOTETomek is a combination of SMOTE and Tomek's link undersampling, which removes examples of the majority class that are close to examples of the minority class. 
This helps to further balance the class distribution and remove potential noisy data points that may have a negative impact on the model's performance.

For further reading about this, see [here](https://imbalanced-learn.org/)

3. Model training and evaluation: With the prepared data and selected features, we will then train and evaluate a machine learning model to predict customer churn. 
This involves splitting the data into training and test sets, selecting an appropriate model type and hyperparameters, 
and evaluating the model's performance using metrics such as accuracy, precision, recall and f1.

Here, Logistic regression, random forest and XGBoost model are constructed and compared the performance to obtain the optimum model for the churn prediction. 

4. Model optimization and deployment: Hyperparameter tuning for the constructed model are performed via various iteration on the model and features to optimize performance.
 Once the model is satisfactory, it can then be deployed for use in the company's retention efforts.

### Results

The best model for churn prediction using the provided dataset is XGBoost with the following parameters and test scores:
```
params_xgb
{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
```
```
     precision    recall  f1-score   support

           0       0.97      0.99      0.98       730
           1       0.95      0.79      0.86       120

    accuracy                           0.96       850
   macro avg       0.96      0.89      0.92       850
weighted avg       0.96      0.96      0.96       850
```
Confusion matrix:

<img src="/images/cm.png" width="400" align = "center">


## References
* https://www.kaggle.com/c/customer-churn-prediction-2020

## Contributing Members

**Team Leads (Contacts) : [Anh Nguyen ](https://github.com/avtnguyen)**

## Contact
* Feel free to contact team leads with any questions or if you are interested in contributing!

### License
This project is licensed under the MIT License - see the LICENSE file for details.