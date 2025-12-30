# Employee Churn Data Analysis Project

Project Overview
This project analyzes the Employee Churn Data from Kaggle to uncover insights about employee turnover. The goal is to explore patterns, trends, and relationships in the data that help understand why employees leave or stay in a company. (Dataset link
https://www.kaggle.com/datasets/ziya07/employee-churn-data)

Dataset Description

The dataset contains employee records with the following types of information:
Demographics: age, gender, education, etc.
Job-related details: department, job role, tenure, overtime, promotions.
Performance indicators: performance ratings, number of projects, satisfaction levels.
Churn target: indicates whether the employee has left the company.

Objectives

- Perform Exploratory Data Analysis (EDA) to understand employee demographics and job-related factors.
- Visualize key trends in churn across departments, roles, and satisfaction levels.
- Identify potential factors influencing employee turnover.
- Summarize findings in an actionable way for HR or business decision-making.
- Prepare the dataset for ML: handle missing values, encode categorical variables, and scale features if needed.
- Build classification models to predict employee churn (e.g., Logistic Regression, Random Forest, XGBoost).
- Evaluate model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- Analyze feature importance to determine key drivers of churn.
- Summarize patterns and ML findings in actionable recommendations.
- Suggest HR strategies to reduce turnover based on data-driven insights.

KPI:
- Business CLV KPI: Performance Rating, Projects Completed
- Business churn cost KPI: with Tenure: 1 = 0.5 x Salary, 2 = 0.75 x Salary, 3-5 = 1 x Salary, 6-9 > 1.25 x Salary, 
9> 1.5 x Salary, Manager Tenure: <4 1.5 x Salary, >3 2 x Salary
-Business KPI retention is 80% in this dataset

Hypotheses:
- Employees with short tenure (e.g., < 1 year) have a higher probability of churn than employees with longer tenure.
- Churn risk decreases with increasing tenure up to a certain point and then stabilizes.
- Low job satisfaction is significantly associated with higher churn
- Highly engaged employees are less likely to leave the company.
- Employees working overtime are more likely to leave than those who do not work overtime.
- Too many projects increase the risk of churn.
- Younger employees are more likely to churn than older employees.
- Educational level influences the likelihood of leaving the company.
- Not being promoted for a long time increases the risk of churn.
- Employees who were promoted in the last year are less likely to leave.
- There are significant differences in churn between departments.
- Some job roles have significantly higher churn than others.
- The tenure variable is among the top 5 predictors of churn.