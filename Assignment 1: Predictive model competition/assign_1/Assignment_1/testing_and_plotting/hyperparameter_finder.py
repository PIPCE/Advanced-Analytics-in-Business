from ..boost import cv_xgboost,cv_csxgboost



def hyperpar_finder(methods,
        covariates, fraud_indicators,  fold, repeats):

    dict = {}

    if 'XGBoost' in methods:

        opt_max_depth,opt_n_estimators,opt_lamb_xg,opt_dropout,opt_learning_rate = cv_xgboost(covariates, fraud_indicators,
                             [5,10], [50,100,500], [0,0.1,1],[0.1, 0.25, 0.4], [0.01],fold, repeats)

        dict.update({"opt_max_depth": opt_max_depth,
                     "opt_n_estimators": opt_n_estimators,
                     "opt_lamb_xg": opt_lamb_xg,
                     "opt_dropout": opt_dropout,
                     "opt_learning_rate:": opt_learning_rate})

    if 'CS_XGBoost' in methods:

        cs_opt_max_depth,cs_opt_n_estimators,cs_opt_lambd_xg,cs_opt_dropout,opt_learning_rate = cv_csxgboost(covariates, fraud_indicators,
                             [5,10], [50,100, 500], [0,0.1,1],[0.1, 0.25, 0.4], [0.01],fold, repeats)

        dict.update({"cs_opt_max_depth": cs_opt_max_depth,
                     "cs_opt_n_estimators": cs_opt_n_estimators,
                     "cs_opt_lambd_xg": cs_opt_lambd_xg,
                     "cs_opt_dropout": cs_opt_dropout,
                     "opt_learning_rate:": opt_learning_rate})

    return dict
