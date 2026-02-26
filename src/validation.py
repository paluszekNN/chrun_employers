import numpy as np
import pandas as pd


def get_k_best_data(x, y, model, k=100):
    y_score = model.predict_proba(x)[:,1]
    y = np.asarray(y)

    top_k_idx = np.argsort(y_score)[::-1][:k]

    x= x.reset_index(drop=True).loc[top_k_idx]
    y=y[top_k_idx]
    return x, y

def get_income(x, y, model):
    x_k, y_k = get_k_best_data(x, y, model)
    x_k['Churn'] = y
    x_k['Cost of outflow'] = 0
    less_then_3_tenure_mask = x_k['Tenure']<3
    x_k['Cost of outflow'].loc[less_then_3_tenure_mask] = (x_k.loc[less_then_3_tenure_mask]['Salary']/12+x_k.loc[less_then_3_tenure_mask]['Salary']/12/x_k.loc[less_then_3_tenure_mask]['Average Monthly Hours Worked']*x_k.loc[less_then_3_tenure_mask]['Training Hours'])*x_k.loc[less_then_3_tenure_mask, 'Churn']
    less_then_9_tenure_mask = x_k['Tenure']<9
    x_k['Cost of outflow'].loc[less_then_9_tenure_mask] = (x_k.loc[less_then_9_tenure_mask]['Salary']/12*2+x_k.loc[less_then_9_tenure_mask]['Salary']/12/x_k.loc[less_then_9_tenure_mask]['Average Monthly Hours Worked']*x_k.loc[less_then_9_tenure_mask]['Training Hours'])*x_k.loc[less_then_9_tenure_mask, 'Churn']
    greater_then_8_tenure_mask = x_k['Tenure']>8
    x_k['Cost of outflow'].loc[greater_then_8_tenure_mask] = (x_k.loc[greater_then_8_tenure_mask]['Salary']/12*3+x_k.loc[greater_then_8_tenure_mask]['Salary']/12/x_k.loc[greater_then_8_tenure_mask]['Average Monthly Hours Worked']*x_k.loc[greater_then_8_tenure_mask]['Training Hours'])*x_k.loc[greater_then_8_tenure_mask, 'Churn']

    x_k['Cost of intervention'] = 5000
    x_k['score'] = x_k['Cost of outflow'] - x_k['Cost of intervention']
    income = x_k['score'].sum()
    print(f"income = {income}")

    ROI = x_k['score'].sum()/x_k['Cost of intervention'].sum()
    print(f"ROI= {round(ROI*100,2)}%")


def get_SHAP(features, model):
    SHAP = pd.Series(model.coef_[0], index=features)
    SHAP['bias'] = model.intercept_[0]
    print('SHAP:')
    print(SHAP.sort_values())
