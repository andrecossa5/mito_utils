"""
Supervised.py: functions for classification models training and evaluation. 
"""

import numpy as np
import shap
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.metrics import *
from tune_sklearn import TuneGridSearchCV


##


def classification(X, y, key='logit', GS=True, n_combos=50, score='f1', cores_model=8, 
                cores_GS=1, GS_mode='bayes', full_output=False, feature_names=None):
    """
    Given some input data X y, run a classification analysis in several flavours.
    """

    ########### Standard sklearn models
    models = {

        'logit' : 
        LogisticRegression(solver='saga', penalty='elasticnet', n_jobs=cores_model, max_iter=10000),
        
        'xgboost' : 
        LGBMClassifier(n_jobs=cores_model, learning_rate=0.1),
        
        'kNN' :
        KNeighborsClassifier(n_jobs=cores_model)

    }
    params = {

        'logit' : 
        {
            'logit__C' : [100, 10, 1.0, 0.1, 0.01],
            'logit__l1_ratio' : np.linspace(0, 1, 10)
        },
        
        'xgboost' : 
        {
            "xgboost__num_leaves" : np.arange(20, 3000, 600),
            "xgboost__n_estimators" : np.arange(100, 600, 100),
            "xgboost__max_depth" : np.arange(3, 12, 2)
        },

        'SVM':
        {
            "SVM__gamma" : [0.01, 0.1, 1, 10, 100],
            "SVM__C" : [0.1, 1, 10, 100, 1000]
        },

        'kNN' :
        {
            "kNN__n_neighbors" : np.arange(5, 100, 25),
            "kNN__metric" : ['cosine', 'l2', 'euclidean']
        }

    }
    ###########

    # Train-test split 
    seed = 1234
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=seed)

    if issparse(X):
        X = X.A # Densify if genes as features

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Pipe or hyperopt-model definition
    if GS_mode == 'random' and GS:

        # Pipeline definiton
        pipe = Pipeline( 
            steps=[ 

                ('pp', StandardScaler()), # Always scale expression features
                (key, models[key])
            ]
        )
        # GS definition
        model = RandomizedSearchCV(
            pipe, 
            param_distributions=params[key], 
            n_iter=n_combos,
            refit=True,
            n_jobs=cores_GS,
            scoring=score,
            random_state=seed,
            cv=StratifiedShuffleSplit(n_splits=5),
            verbose=True
        )

        # Fit and find best model
        model.fit(X_train, y_train)
        f = model.best_estimator_[key]

    elif GS_mode == 'bayes' and GS:

        # Pipeline definiton
        pipe = Pipeline( 
            steps=[ 

                ('pp', StandardScaler()), # Always scale expression features
                (key, models[key]) 
            ]
        )

        # Ray-tune choice and training 
        model = TuneGridSearchCV(
            pipe,
            params[key],
            scoring=score,
            refit=True,
            n_jobs=cores_GS,
            cv=StratifiedShuffleSplit(n_splits=5),
            early_stopping=False,
            max_iters=n_combos
        )
        model.fit(X_train, y_train)
        f = model.best_estimator_[key]

    else:
        model = pipe
        model.fit(X_train, y_train)
        f = model[key]

    # Decision treshold tuning
    precisions, recalls, tresholds = precision_recall_curve(
        y_train, f.predict_proba(X_train)[:,1], 
    )
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    alpha = tresholds[np.argmax(f1_scores)]

    # Testing
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    y_pred_probabilities = f.predict_proba(X_test)[:,1]
    y_pred = [ 1 if y >= alpha else 0 for y in y_pred_probabilities ]

    d = {
        'accuracy' : accuracy_score(y_test, y_pred),
        'balanced_accuracy' : balanced_accuracy_score(y_test, y_pred),
        'precision' : precision_score(y_test, y_pred),
        'recall' : recall_score(y_test, y_pred),
        'f1' : f1_score(y_test, y_pred)
    }

    if full_output and feature_names is not None:
        
        explainer = shap.Explainer(model.predict, X_test, feature_names=feature_names)
        SHAP = explainer(X_test)

        results = {
            'best_estimator' : f,
            'performance_dict': d, 
            'SHAP' : SHAP, 
            'y_test' : y_test, 
            'y_pred' : y_pred, 
            'precisions' : precisions, 
            'recalls' : recalls,
            'tresholds' : tresholds,
            'alpha' : alpha  
        }
        
        return results
    
    else:
        return d
        

##