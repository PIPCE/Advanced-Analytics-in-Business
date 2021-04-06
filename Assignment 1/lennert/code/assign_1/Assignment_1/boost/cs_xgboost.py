from timeit import default_timer as timer
import numpy as np
import xgboost as xgb


# class CSXgboost:
#
#     def __init__(self, n_estimators, max_depth, lambd, dropout, learning_rate, fixed_cost = 1):
#
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.learning_rate =learning_rate
#         self.lambd = lambd
#         self.dropout = dropout
#         self.fixed_cost = fixed_cost
#
#
#     def predict(self, model, X_test, treshhold):
#
#         dtest = xgb.DMatrix(X_test)
#         scores = model.predict(dtest)
#         predictions = (scores > treshhold).astype(int)
#
#         return predictions
#
#     def predict_proba(self, model, X_test):
#
#         dtest = xgb.DMatrix(X_test)
#         scores = model.predict(dtest)
#
#         return scores
#
#     def fitting(self, X, y, cost):
#
#         starttimer = timer()
#
#         #weight = np.where(y == 1, cost, self.fixed_cost)
#         #weight = np.where(y == 1, np.log(cost), self.fixed_cost)
#
#         # test
#         fraud_amounts = np.where(y == 1, cost, 0)
#         mean_fraud_amounts = fraud_amounts[np.nonzero(fraud_amounts)].mean()
#         #print(mean_fraud_amounts)
#         #cost[cost < mean_fraud_amounts] = mean_fraud_amounts
#         #print(np.max(cost))
#         weight = np.where(y == 1,cost/mean_fraud_amounts, 1)
#
#
#         dtrain = xgb.DMatrix(X, label=y, weight=weight)
#
#         param = {'max_depth': self.max_depth,  'lambda' : self.lambd, 'subsample':self.dropout,
#                  'eta': self.learning_rate, 'objective': 'binary:logistic'}
#         n_estimators = self.n_estimators
#
#         model = xgb.train(param, dtrain = dtrain, num_boost_round=n_estimators)
#
#         endtimer = timer()
#
#         return model, endtimer-starttimer


class CSXgboost:

    def __init__(self, n_estimators, max_depth, lambd, dropout, learning_rate, fixed_cost=1):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate =learning_rate
        self.fixed_cost = fixed_cost
        self.lambd = lambd
        self.dropout = dropout


    def predict(self, model, X_test, treshhold):

        dtest = xgb.DMatrix(X_test)
        scores = model.predict(dtest)

        scores = 1 / (1 + np.exp(-scores))
        predictions = (scores > treshhold).astype(int)

        return predictions

    def predict_proba(self, model, X_test):

        dtest = xgb.DMatrix(X_test)
        scores = model.predict(dtest)
        scores = 1 / (1 + np.exp(-scores))

        return scores

    def fitting(self, X, y, amount):

        starttimer = timer()

        # div = 1 / X.shape[0]
        int_value = y


        fraud_amounts = np.where(y == 1, amount, 0)
        mean_fraud_amounts = fraud_amounts[np.nonzero(fraud_amounts)].mean()
        #print(mean_fraud_amounts)
        amount[amount < mean_fraud_amounts] = mean_fraud_amounts
        #print(np.max(cost))
        self.fixed_cost = mean_fraud_amounts

        def objective_function(predt, dtrain):

            #objective value = average expected cost
            scores1 = 1 / (1 + np.exp(-predt))

            # objective = div * (int_value.dot(1-scores1) +
            #                     np.sum(scores1)*self.fixed_cost)    #eq 7
            # print(objective)
            y = dtrain.get_label()

            a= (1-y)*self.fixed_cost - np.multiply(amount,y)

            grad = np.multiply(np.multiply(scores1,1-scores1),a)

            hess= np.abs(np.multiply(grad, 1-(2*scores1) ) ) # abs: pos definite get minimum

            return grad, hess

        #dtrain = xgb.DMatrix(X, label=int_value)

        #https://xgboost.readthedocs.io/en/latest/parameter.html  extra parameters?

        #param = {'max_depth': self.max_depth,'subsample':self.dropout,
        #         'eta': self.learning_rate, 'lambda' : self.lambd}
        #
        n_estimators = self.n_estimators

        #model = xgb.train(param, dtrain = dtrain, num_boost_round=n_estimators, obj=objective_function)

        weight = np.where(y == 1, amount / mean_fraud_amounts, 1)
        dtrain = xgb.DMatrix(X, label=y, weight=weight)

        param = {'max_depth': self.max_depth,  'lambda' : self.lambd, 'subsample':self.dropout,
                          'eta': self.learning_rate, 'objective': 'binary:logistic'}

        model = xgb.train(param, dtrain = dtrain, num_boost_round=n_estimators)


        endtimer = timer()

        return model, endtimer-starttimer




