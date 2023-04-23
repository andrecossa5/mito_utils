"""
Supervised.py: functions for classification models training and evaluation. 
"""

import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.metrics import *

from hyperopt import tpe, hp
from hpsklearn import (
                            HyperoptEstimator, 
                            k_neighbors_classifier, 
                            standard_scaler, 
                            logistic_regression, 
                            lightgbm_classification
                        )


##


def classification(X, y, key='logit', GS=True, n_combos=50, score='f1', cores_model=8, cores_GS=1, GS_mode='bayes'):
    """
    Given some input data X y, run a classification analysis in several flavours.
    """

    ########### Standard sklearn models
    models = {

        'logit' : 
        LogisticRegression(solver='saga', penalty='elasticnet', n_jobs=cores_model, max_iter=1000),
        
        'xgboost' : 
        LGBMClassifier(n_jobs=cores_model, learning_rate=0.1),
        
        'kNN' :
        KNeighborsClassifier(n_jobs=cores_model)

    }
    ###########
    ##
    ########### Sklarn GS spaces
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

    ##

    ########### Hyperoptim models
    hyper_models = {

        'logit' : logistic_regression('logistic_regression'),
                #     'logistic_regression', solver='saga', penalty='elasticnet', 
                #     n_jobs=cores_model, max_iter=1000, 
                #     C=hp.choice('C', [100, 10, 1.0, 0.1, 0.01]),
                #     l1_ratio=hp.choice('l1_ratio', np.linspace(0, 1, 10))
                # ),

        'xgboost' : lightgbm_classification('xgboost'),
                #     'lightgbm_classification', learning_rate=0.1,
                #     num_leaves=hp.choice('num_leaves', np.arange(20, 3000, 600)),
                #     n_estimators=hp.choice('n_estimators', np.arange(100, 600, 100)),
                #     max_depth=hp.choice('max_depth', np.arange(3, 12, 2))
                # ),

        'kNN' : k_neighbors_classifier('k_neighbors_classifier')
                #     'k_neighbors_classifier', 
                #     n_jobs=cores_model,
                #     n_neighbors=hp.choice('n_neighbors', np.arange(5, 100, 25)),
                #     metric=hp.choice('metric', ['cosine', 'l2', 'euclidean'])
                # )
    
    }
    ###########


    ##

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

        # Pipelinee definiton
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

        # Hyperoptim model choice and training 
        model = HyperoptEstimator(
                    classifier=hyper_models[key], 
                    preprocessing=[standard_scaler(name='standard_scaler', with_mean=True, with_std=True)],
                    algo=tpe.suggest,
                    max_evals=n_combos,
                    loss_fn=f1_score, 
                    trial_timeout=120,
                    refit=True,
                    n_jobs=cores_model,
                    seed=seed
                )
        
        # Find best model
        model.fit(X_train, y_train)
        f = model._best_learner

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

    return d


##