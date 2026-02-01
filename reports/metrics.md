# models score (auc, precision@100)

| Model                                               | AUC  | Precision@100 |
|-----------------------------------------------------|------|---------------|
| DummyClassifier(strategy='most_frequent')           | 0.5  | 0.21          |
| LogisticRegression()                                | 0.50 | 0.24          |
| LogisticRegression() with feature engineering       | 0.52 | 0.23          |
| RandomForestClassifier() with feature engineering   | 0.5  | 0.18          |
| XGBClassifier() with feature engineering            | 0.5  | 0.22          |
| XGBClassifier() with feature engineering and tuning | 0.48 | 0.243         |

Best model chosen for simulation is XGBClassifier() with feature engineering and tuning because of high precision at 100 as simulation will take 100 employers.