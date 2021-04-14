from ..boost import cv_xgboost,cv_csxgboost


def hyperpar_finder(methods,
        covariates, fraud_indicators, cost, fixed_cost,  fold, repeats):

    dict = {}


    if 'XGBoost' in methods:
        opt_max_depth, opt_n_estimators, opt_lamb_xg, opt_dropout, opt_learning_rate = cv_xgboost(covariates,
                 fraud_indicators, cost,fixed_cost, [5], [1000], [0.1,1,3,5],[0.3], [0.01],
                                                                                               fold,
                                                                                              repeats)
        dict.update({"xg_opt_max_depth": opt_max_depth,
                     "xg_opt_n_estimators": opt_n_estimators,
                     "xg_opt_lamb": opt_lamb_xg,
                     "xg_opt_dropout": opt_dropout,
                     "xg_opt_learning_rate": opt_learning_rate})

    if 'CS_XGBoost' in methods:
        cs_opt_max_depth, cs_opt_n_estimators, cs_opt_lambd_xg, cs_opt_dropout, cs_opt_learning_rate = cv_csxgboost(
            covariates, fraud_indicators, cost, fixed_cost,
            [5], [1000], [0.1,1,3,5],[0.3], [0.01],  fold, repeats)

        dict.update({"cs_xg_opt_max_depth": cs_opt_max_depth,
                     "cs_xg_opt_n_estimators": cs_opt_n_estimators,
                     "cs_xg_opt_lambd": cs_opt_lambd_xg,
                     "cs_xg_opt_dropout": cs_opt_dropout,
                     "cs_xg_opt_learning_rate": cs_opt_learning_rate})


    return dict

# def hyperpar_finder(methods,cost,
#         covariates, fraud_indicators,  fold, repeats):
#
#     dict = {}
#
#     if 'XGBoost' in methods:
#
#         opt_max_depth,opt_n_estimators,opt_lamb_xg,opt_dropout,opt_learning_rate = cv_xgboost(covariates, fraud_indicators,cost,
#                              [5,10], [50, 500, 1000], [0,0.01],[0.1, 0.4], [0.01, 0.1],fold, repeats)
#
#         dict.update({"opt_max_depth": opt_max_depth,
#                      "opt_n_estimators": opt_n_estimators,
#                      "opt_lamb_xg": opt_lamb_xg,
#                      "opt_dropout": opt_dropout,
#                      "opt_learning_rate:": opt_learning_rate})
#
#     if 'CS_XGBoost' in methods:
#
#         cs_opt_max_depth,cs_opt_n_estimators,cs_opt_lambd_xg,cs_opt_dropout,cs_opt_learning_rate = cv_csxgboost(covariates, fraud_indicators,cost,
#                              [5,10], [50,500, 1000], [0,0.01],[0.1,0.4], [0.01,0.1],fold, repeats)
#
#         dict.update({"cs_opt_max_depth": cs_opt_max_depth,
#                      "cs_opt_n_estimators": cs_opt_n_estimators,
#                      "cs_opt_lambd_xg": cs_opt_lambd_xg,
#                      "cs_opt_dropout": cs_opt_dropout,
#                      "cs_opt_learning_rate:": cs_opt_learning_rate})
#
#     return dict
