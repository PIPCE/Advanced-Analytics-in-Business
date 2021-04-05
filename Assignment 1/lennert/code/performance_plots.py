import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..boost import Xgbst, CSXgboost
from ..design_and_metrics import PerformanceMetrics
from sklearn.model_selection import RepeatedStratifiedKFold


def performance_plot(methods, opt_par_dict, covariates, clas, fold, repeats):

    if 'XGBoost' in methods:

        opt_max_depth = opt_par_dict.get("opt_max_depth")
        opt_n_estimators = opt_par_dict.get("opt_n_estimators")
        opt_lamb_xg = opt_par_dict.get("opt_lamb_xg")
        opt_dropout_xg = opt_par_dict.get("opt_dropout")
        opt_learning_rate_xg = opt_par_dict.get("opt_learning_rate")
        XGBoost_F_list = []

    if 'CS_XGBoost' in methods:

        cs_opt_max_depth = opt_par_dict.get("cs_opt_max_depth")
        cs_opt_n_estimators = opt_par_dict.get("cs_opt_n_estimators")
        cs_opt_lambd_xg = opt_par_dict.get("cs_opt_lambd_xg")
        cs_opt_dropout_xg = opt_par_dict.get("cs_opt_dropout")
        cs_opt_learning_rate_xg =opt_par_dict.get("cs_opt_learning_rate")
        CS_XGBoost_F_list = []

    covariates = np.array(covariates)
    clas = clas.values
    rskf = RepeatedStratifiedKFold(n_splits=fold, n_repeats=repeats, random_state=2290)

    i = 0
    for train_index, test_index in rskf.split(covariates, clas):
        print('Cross Validation number ' + str(i))

        X_train, X_test = covariates[train_index], covariates[test_index]
        y_train, y_test = clas[train_index], clas[test_index]

        if 'XGBoost' in methods:

            F_xgboost = \
                xgboost_ratio(opt_max_depth, opt_n_estimators, opt_lamb_xg,opt_dropout_xg,opt_learning_rate_xg,
                                       X_train, y_train, X_test, y_test)

            XGBoost_F_list.append(F_xgboost)

        if 'CS_XGBoost' in methods:

            F_cs_xgboost = cs_xgboost_ratio(cs_opt_max_depth,
                                                cs_opt_n_estimators, cs_opt_lambd_xg,cs_opt_dropout_xg,cs_opt_learning_rate_xg,
                                             X_train, y_train, X_test, y_test)

            CS_XGBoost_F_list.append(F_cs_xgboost)

        i += 1

    F_df= pd.DataFrame()

    if 'XGBoost' in methods:
        F_df['XGBoost'] = XGBoost_F_list
        print("F-value xgboost (%) " + str(format(np.mean(XGBoost_F_list), '.4f')))

    if 'CS_XGBoost' in methods:
        F_df['CS_XGBoost'] = CS_XGBoost_F_list
        print("F-Value cs xgboost (%) " + str(format(np.mean(CS_XGBoost_F_list), '.4f')))

    plt.xlabel('Methods')
    plt.ylabel('F-value')
    sns.boxplot(data=F_df)
    plt.show()


def xgboost_ratio(opt_max_depth, opt_n_estimators, opt_lambd_xg, opt_dropout_xg,opt_lr,
                  X_train, y_train, X_test,
                 y_test):

    xgboost = Xgbst(n_estimators =opt_n_estimators, max_depth = opt_max_depth,
                    lambd = opt_lambd_xg, dropout= opt_dropout_xg, learning_rate = opt_lr)
    xgbst_train, time = xgboost.fitting(X_train,y_train)

    xgpred = xgboost.predict_proba(xgbst_train, X_test)

    thresholds = np.arange(0, 1, 0.01)
    def to_labels(pos_probs, threshold):
	    return (pos_probs >= threshold).astype('int')

    scores = [PerformanceMetrics.confusion_matrix(y_test,
                                                to_labels(xgpred, t)) for t in thresholds]

    ix = np.nanargmax(scores)
    xgpred = xgboost.predict(xgbst_train, X_test,thresholds[ix])

    F_value = PerformanceMetrics.confusion_matrix(xgpred, y_test)
    return F_value

def cs_xgboost_ratio(opt_max_depth, opt_n_estimators, opt_lambd_xg,opt_dropout_xg,opt_lr,
                      X_train, y_train, X_test, y_test ):

    csxgboost = CSXgboost(n_estimators =opt_n_estimators, max_depth = opt_max_depth,
                          lambd = opt_lambd_xg, dropout= opt_dropout_xg, learning_rate = opt_lr)
    xgbst_train, time = csxgboost.fitting(X_train,y_train)

    xgpred = csxgboost.predict_proba(xgbst_train, X_test)

    thresholds = np.arange(0, 1, 0.01)
    def to_labels(pos_probs, threshold):
	    return (pos_probs >= threshold).astype('int')

    scores = [PerformanceMetrics.confusion_matrix(y_test,
                                                to_labels(xgpred, t)) for t in thresholds]

    ix = np.nanargmax(scores)
    xgpred = csxgboost.predict(xgbst_train, X_test,thresholds[ix])

    F_value = PerformanceMetrics.confusion_matrix(xgpred, y_test)
    return F_value
