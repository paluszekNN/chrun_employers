import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('..\\data\\employee_churn_dataset.csv')
for column in df.drop('Employee ID', axis=1).columns:
    try:
        df[column] + 1
    except:
        dummies = pd.get_dummies(df[column]).add_prefix(column + '_')
        df[dummies.columns] = dummies
        df.drop(column, axis=1, inplace=True)
features = df.drop(['Churn', 'Employee ID'], axis=1).columns
data_X = df[features]
data_y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)