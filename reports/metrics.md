# models score (auc, precision@100)

| Model                                             | AUC | Precision@100 |
|---------------------------------------------------|-----|---------------|
| DummyClassifier(strategy='most_frequent')         | 0.5 | 0.21          |
| LogisticRegression()                              | 0.5 | 0.2           |
| LogisticRegression() with feature engineering     | 0.5 | 0.19          |
| RandomForestClassifier() with feature engineering | 0.5 | 0.18          |
| XGBClassifier() with feature engineering          | 0.5 | 0.15          |