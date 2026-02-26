from validation import *
from modeling import *
from data import X_train, X_test, y_train, y_test

if __name__ == '__main__':
    results = get_results_from_grid(X_train, y_train)
    print("top 5 best models:")
    for i in range(5):
        print(f"params: {results['params'].iloc[i]}")
        print(f"mean_test_score: {results['mean_test_p_at_k'].iloc[i]}")
        print(f"std_test_score: {results['std_test_p_at_k'].iloc[i]}")

    print('Dummy model AUC:')
    print(results.loc[results['param_features'].isna()]['mean_test_auc'].iloc[0])
    print('Best model AUC:')
    print(results.loc[0]['mean_test_auc'])
    model = LogisticRegression()
    model.fit(X_train, y_train)
    get_income(X_test,y_test,model)
    get_SHAP(X_test.columns,model)
