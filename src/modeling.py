from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
from src.feature_engineering import FeatureEngineering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import pandas as pd


def precision_at_k(estimator, X, y, k=100):
    y_score = estimator.predict_proba(X)[:,1]
    y = np.asarray(y)

    top_k_idx = np.argsort(y_score)[::-1][:k]

    return y[top_k_idx].sum() / k

def validate_model(model, X_test, y_test):
    y_score = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_score)
    precision_at_k_score = precision_at_k(model, X_test, y_test, 100)
    return auc_score, precision_at_k_score

def get_coef(estimator, X_train):
    coef = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient":estimator.coef_.ravel()
                  })
    coef = coef.sort_values("coefficient", ascending=False)
    return coef

def get_results_from_grid(X_train, y_train):
    pipeline = Pipeline(steps=[('features', FeatureEngineering()),
                               ('scaler', 'passthrough'),
                               ('pca', 'passthrough'),
                               ('model', LogisticRegression())])

    param_grid = [
        {'model': [DummyClassifier(strategy='most_frequent')]},
        {
            'pca': ['passthrough', PCA(n_components=10), PCA(n_components=30)],
            'features': ['passthrough', FeatureEngineering()],
            'scaler': ['passthrough', StandardScaler()],
            'model': [LogisticRegression(
                solver="saga",
                max_iter=2000,
                n_jobs=1,
                random_state=42)],
            "model__penalty": ["l1", "l2"],
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"]
        },

        {
            'scaler': ['passthrough'],
            'pca': ['passthrough'],
            'model': [RandomForestClassifier(
                random_state=42, n_estimators=300, n_jobs=1, class_weight="balanced"
            )],
            'model__max_depth': [None, 8, 12, 16],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 5, 10, 20],
            'model__max_features': ["sqrt", 0.3, 0.5]
        },

        {
            'scaler': ['passthrough'],
            'pca': ['passthrough'],
            'model': [XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                objective="binary:logistic",
                scale_pos_weight=0.2,
                eval_metric=precision_at_k,
                random_state=42,
                n_jobs=1
            )],
            'model__max_depth': [3, 4, 5, 6],
            'model__eta': [0.01, 0.1, 0.3],
            'model__min_child_weight': [1, 3, 5, 7],
            'model__subsample': [0.7, 0.8, 0.9],
            'model__colsample_bytree': [0.7, 0.8, 0.9],
            'model__gamma': [0, 0.5, 1],
            'model__reg_alpha': [1, 2, 5],
        }
    ]
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
            "auc": "roc_auc",
            "p_at_k": precision_at_k
        },
        refit='p_at_k',
        cv=cv,
        n_jobs=-4,
        verbose=3
    )

    grid.fit(X_train, y_train)
    results = pd.DataFrame(grid.cv_results_)
    return results
