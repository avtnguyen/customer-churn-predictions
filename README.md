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

To run the analysis, follow the steps in the Jupyter Notebook. The notebook includes data exploration, preprocessing, and model training and evaluation.

**Built With**

* Python - Programming language

* Jupyter Notebook - Interactive coding environment

* pandas - Data manipulation and analysis library

* scikit-learn - Machine learning library

* xgboost - Gradient boosting algorithm

* matplotlib and searborn - Data visualization

### Project pipeline
1. Data acquisition and preprocessing: 
The first step in the project is to obtain and prepare the data for analysis. 
This involves collecting data from multiple sources, cleaning and formatting the data, and performing any necessary preprocessing steps.
'''
There are 19 features: ['state', 'account_length', 'area_code', 'international_plan', 'voice_mail_plan', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls', 'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls']

The label is churn that takes two values no: customer not leaving, and yes: customer leaving with the following summary
Train dataset:
 no     3652
yes     598
Name: churn, dtype: int64

'''
<img src="/images/hisplot.png" width="400" align = "center">

## References:
* https://www.kaggle.com/c/customer-churn-prediction-2020

## Contributing Members

**Team Leads (Contacts) : [Anh Nguyen ](https://github.com/avtnguyen)**

## Contact
* Feel free to contact team leads with any questions or if you are interested in contributing!

### License
This project is licensed under the MIT License - see the LICENSE file for details.