import numpy as np
from ..design_and_metrics import PerformanceMetrics
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from .cs_xgboost import CSXgboost
from .xgbst import Xgbst

def cv_xgboost(covariates, clas,max_depth_list, n_estimators_list,
               lambd_list,opt_dropout_list,learning_rate_list,
                fold, repeats):

    covariates = np.array(covariates)
    clas = clas.values

    rskf = RepeatedStratifiedKFold(n_splits=fold, n_repeats=repeats,random_state=2290)
    F_measure_matrix = np.zeros(shape=(len(max_depth_list),
                                        len(n_estimators_list),
                                        len(lambd_list),
                                        len(opt_dropout_list),
                                       len(learning_rate_list)), dtype='object')

    for i, max_depth in enumerate(max_depth_list):
        for j, n_estimators in enumerate(n_estimators_list):
            for k, lambd in enumerate(lambd_list):
                for l, dropout in enumerate(opt_dropout_list):
                    for m, learning_rate in enumerate(learning_rate_list):
                        print('Cross Validation for max_depth equals '+ str(max_depth))
                        print('Cross Validation for n_estimators equals '+ str(n_estimators))
                        print('Cross Validation for lambd equals '+ str(lambd))
                        print('Cross Validation for dropout equals '+ str(dropout))
                        print('Cross Validation for learning rate equals ' + str(learning_rate))

                        for train_index, test_index in rskf.split(covariates, clas):

                            X_train, X_test = covariates[train_index], covariates[test_index]
                            y_train, y_test = clas[train_index], clas[test_index]

                            xgboost = Xgbst(n_estimators =n_estimators, max_depth = max_depth,
                                                   lambd= lambd,dropout=dropout,learning_rate = learning_rate)
                            xgbst_train, time = xgboost.fitting(X_train,y_train)
                            xgpred = xgboost.predict_proba(xgbst_train, X_test)

                            thresholds = np.arange(0, 1, 0.01)
                            def to_labels(pos_probs, threshold):
                                return (pos_probs >= threshold).astype('int')

                            scores = [PerformanceMetrics.confusion_matrix(y_test,
                                                    to_labels(xgpred, t)) for t in thresholds]

                            ix = np.nanargmax(scores)
                            xgpred_opt = xgboost.predict(xgbst_train, X_test,thresholds[ix])

                            F_measure = PerformanceMetrics.confusion_matrix(xgpred_opt,y_test)

                            F_measure_matrix[i,j,k] += F_measure


    print(F_measure_matrix)
    max_indices =  np.unravel_index(F_measure_matrix.argmax(), np.shape(F_measure_matrix))
    opt_max_depth = max_depth_list[max_indices[0]]
    opt_n_estimators = n_estimators_list[max_indices[1]]
    opt_lambd = lambd_list[max_indices[2]]
    opt_dropout = opt_dropout_list[max_indices[3]]
    opt_learning_rate = learning_rate_list[max_indices[4]]
    print('The optimal depth equals ' + str(opt_max_depth))
    print('The optimal n_estimators equals ' + str(opt_n_estimators))
    print('The optimal lambd equals ' + str(opt_lambd))
    print('The optimal opt_dropout equals ' + str(opt_dropout))
    print('The optimal learning rate equals ' + str(opt_learning_rate))

    return opt_max_depth,opt_n_estimators,opt_lambd,opt_dropout,opt_learning_rate


def cv_csxgboost(covariates, clas,max_depth_list, n_estimators_list,
               lambd_list,opt_dropout_list,learning_rate_list,
                fold, repeats):

    covariates = np.array(covariates)
    clas = clas.values

    rskf = RepeatedStratifiedKFold(n_splits=fold, n_repeats=repeats,random_state=2290)
    F_measure_matrix = np.zeros(shape=(len(max_depth_list),
                                        len(n_estimators_list),
                                        len(lambd_list),
                                        len(opt_dropout_list),
                                       len(learning_rate_list)), dtype='object')

    for i, max_depth in enumerate(max_depth_list):
        for j, n_estimators in enumerate(n_estimators_list):
            for k, lambd in enumerate(lambd_list):
                for l, dropout in enumerate(opt_dropout_list):
                    for m, learning_rate in enumerate(learning_rate_list):
                        print('Cross Validation for max_depth equals '+ str(max_depth))
                        print('Cross Validation for n_estimators equals '+ str(n_estimators))
                        print('Cross Validation for lambd equals '+ str(lambd))
                        print('Cross Validation for dropout equals '+ str(dropout))
                        print('Cross Validation for learning rate equals ' + str(learning_rate))

                        for train_index, test_index in rskf.split(covariates, clas):

                            X_train, X_test = covariates[train_index], covariates[test_index]
                            y_train, y_test = clas[train_index], clas[test_index]

                            xgboost = CSXgboost(n_estimators =n_estimators, max_depth = max_depth,
                                                   lambd= lambd,dropout=dropout,learning_rate = 0.01)
                            xgbst_train, time = xgboost.fitting(X_train,y_train)

                            xgpred = xgboost.predict_proba(xgbst_train, X_test)

                            thresholds = np.arange(0, 1, 0.01)
                            def to_labels(pos_probs, threshold):
                                return (pos_probs >= threshold).astype('int')

                            scores = [PerformanceMetrics.confusion_matrix(y_test,
                                                    to_labels(xgpred, t)) for t in thresholds]

                            ix = np.nanargmax(scores)

                            xgpred_opt = xgboost.predict(xgbst_train, X_test,thresholds[ix])

                            F_measure = PerformanceMetrics.confusion_matrix(xgpred_opt,y_test)

                            F_measure_matrix[i,j,k] += F_measure


    print(F_measure_matrix)
    max_indices =  np.unravel_index(F_measure_matrix.argmax(), np.shape(F_measure_matrix))
    opt_max_depth = max_depth_list[max_indices[0]]
    opt_n_estimators = n_estimators_list[max_indices[1]]
    opt_lambd = lambd_list[max_indices[2]]
    opt_dropout = opt_dropout_list[max_indices[3]]
    opt_learning_rate = learning_rate_list[max_indices[4]]
    print('The optimal depth equals ' + str(opt_max_depth))
    print('The optimal n_estimators equals ' + str(opt_n_estimators))
    print('The optimal lambd equals ' + str(opt_lambd))
    print('The optimal opt_dropout equals ' + str(opt_dropout))
    print('The optimal learning rate equals ' + str(opt_learning_rate))

    return opt_max_depth,opt_n_estimators,opt_lambd,opt_dropout,opt_learning_rate
