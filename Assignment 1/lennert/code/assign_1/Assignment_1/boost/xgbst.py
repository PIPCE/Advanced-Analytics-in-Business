
from timeit import default_timer as timer
import xgboost as xgb


class Xgbst:

    def __init__(self, n_estimators, max_depth, lambd, dropout, learning_rate):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate =learning_rate
        self.lambd = lambd
        self.dropout = dropout



    def predict(self, model, X_test, treshhold):

        dtest = xgb.DMatrix(X_test)
        scores = model.predict(dtest)
        predictions = (scores > treshhold).astype(int)

        return predictions

    def predict_proba(self, model, X_test):

        dtest = xgb.DMatrix(X_test)
        scores = model.predict(dtest)

        return scores

    def fitting(self, X, y):

        starttimer = timer()



        dtrain = xgb.DMatrix(X, label=y)
        param = {'max_depth': self.max_depth,  'lambda' : self.lambd, 'subsample':self.dropout,
                 'eta': self.learning_rate, 'objective': 'binary:logistic'}
        n_estimators = self.n_estimators

        model = xgb.train(param, dtrain = dtrain, num_boost_round=n_estimators)

        endtimer = timer()

        return model, endtimer-starttimer





