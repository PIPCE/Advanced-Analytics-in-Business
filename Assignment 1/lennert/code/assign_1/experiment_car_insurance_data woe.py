import numpy as np
import pandas as pd
import Assignment_1
import data.car_insurance_data.load_car_insurance_data as dataloader
import scipy.stats as st
import Assignment_1.boost.xgbst as xgbst
import Assignment_1.boost.cs_xgboost as cs_xgboost
import Assignment_1.design_and_metrics.data_handler as datahandler
from pathlib import Path
import datetime


#cadfsfdg
"""
Load in the data
"""

car_insurance_data_na_train, car_insurance_data_na_test  = dataloader.import_car_insurance_data()

test_index  =car_insurance_data_na_test.index

fraud_map = {'Y': 1, 'N': 0}
car_insurance_data_na_train['fraud'] = car_insurance_data_na_train['fraud'].map(fraud_map)

fraudulent_claims = car_insurance_data_na_train[car_insurance_data_na_train['fraud'] == 1]
print('The shape of the dataset including empty values equals ' + str(car_insurance_data_na_train.shape))
print('The amount of fraudulent claims in the full dataset equals ' + str(len(fraudulent_claims)))


cost = car_insurance_data_na_train['claim_amount'].values
fraud_indicators = car_insurance_data_na_train['fraud']
car_insurance_data_na_train = car_insurance_data_na_train.drop(['fraud', 'claim_amount'], axis =1)


"""
There is an intersection in test and train set
"""

#test_pol_holders = car_insurance_data_na_test['policy_holder_id']
#train_pol_holders = car_insurance_data_na['policy_holder_id']
#print(pd.Series(list(set(train_pol_holders).intersection(set(test_pol_holders)))))

duplicate_trainer = car_insurance_data_na_train.drop('claim_date_registered', axis=1)
duplicate_tester = car_insurance_data_na_test.drop('claim_date_registered', axis=1)
merged = duplicate_trainer.append(duplicate_tester)
duplicator = merged.duplicated()
#print(duplicator[duplicator==True])   #duplicates
"""
Manage columns
"""

#ID related columns

car_insurance_data_train = car_insurance_data_na_train.copy()
car_insurance_data_test = car_insurance_data_na_test.copy()

training_instances = car_insurance_data_train.shape[0]
car_insurance_data = car_insurance_data_train.append(car_insurance_data_test)


#
car_insurance_data = datahandler.DataHandler.deleter(car_insurance_data,
                       ['claim_vehicle_id', 'policy_holder_id',
                                          'driver_id',
                                          'driver_vehicle_id',
                                          'third_party_1_id',
                                          'third_party_1_vehicle_id',
                                          'third_party_2_id',
                                          'third_party_2_vehicle_id',
                                          'third_party_3_id',
                                          'third_party_3_vehicle_id'])


#other time related columns
car_insurance_data = datahandler.DataHandler.\
    date_ymd_converter(car_insurance_data,
                   ['claim_date_occured','claim_date_registered'])


car_insurance_data= datahandler.DataHandler.\
    date_ym_converter(car_insurance_data,
                  ['claim_vehicle_date_inuse','policy_date_start','policy_date_next_expiry',
                                  'policy_date_last_renewed'])

car_insurance_data =datahandler.DataHandler.\
    date_month_differencer(car_insurance_data,'policy_date_next_expiry','claim_date_occured')

car_insurance_data =datahandler.DataHandler.\
    date_month_differencer(car_insurance_data,'claim_date_registered','claim_date_occured')

car_insurance_data =datahandler.DataHandler.\
    date_month_differencer(car_insurance_data,'policy_date_next_expiry','policy_date_last_renewed')

car_insurance_data =datahandler.DataHandler.\
    date_month_differencer(car_insurance_data,'claim_date_occured','policy_date_start')

car_insurance_data =datahandler.DataHandler.\
    date_month_differencer(car_insurance_data,'claim_date_occured','policy_date_last_renewed')


car_insurance_data = datahandler.DataHandler.deleter(car_insurance_data,
              ['claim_date_occured', 'claim_vehicle_date_inuse',
                                          'policy_date_start',
                                          'policy_date_next_expiry',
                                          'policy_date_last_renewed',
               'claim_date_registered'])
columns = list(car_insurance_data.columns.values)
car_insurance_data = datahandler.DataHandler.create_dummies(car_insurance_data,columns)


car_insurance_data_train1 = car_insurance_data.iloc[:training_instances]
car_insurance_data_test1 = car_insurance_data.iloc[training_instances:]


"""
Weight of evidence encoding
"""


car_insurance_data_train1['policy_coverage_type'] = car_insurance_data_train1['policy_coverage_type'].apply(str)
car_insurance_data_train1['claim_postal_code'] = car_insurance_data_train1['claim_postal_code'].apply(str)
car_insurance_data_train1['policy_holder_postal_code'] = car_insurance_data_train1['policy_holder_postal_code'].apply(str)
car_insurance_data_train1['driver_postal_code'] = car_insurance_data_train1['driver_postal_code'].apply(str)
car_insurance_data_train1['third_party_1_postal_code'] = car_insurance_data_train1['third_party_1_postal_code'].apply(str)
car_insurance_data_train1['third_party_2_postal_code'] = car_insurance_data_train1['third_party_2_postal_code'].apply(str)
car_insurance_data_train1['third_party_3_postal_code'] = car_insurance_data_train1['third_party_3_postal_code'].apply(str)
car_insurance_data_train1['repair_postal_code'] = car_insurance_data_train1['third_party_3_postal_code'].apply(str)

car_insurance_data_test1['policy_coverage_type'] = car_insurance_data_test1['policy_coverage_type'].apply(str)
car_insurance_data_test1['claim_postal_code'] = car_insurance_data_test1['claim_postal_code'].apply(str)
car_insurance_data_test1['policy_holder_postal_code'] = car_insurance_data_test1['policy_holder_postal_code'].apply(str)
car_insurance_data_test1['driver_postal_code'] = car_insurance_data_test1['driver_postal_code'].apply(str)
car_insurance_data_test1['third_party_1_postal_code'] = car_insurance_data_test1['third_party_1_postal_code'].apply(str)
car_insurance_data_test1['third_party_2_postal_code'] = car_insurance_data_test1['third_party_2_postal_code'].apply(str)
car_insurance_data_test1['third_party_3_postal_code'] = car_insurance_data_test1['third_party_3_postal_code'].apply(str)
car_insurance_data_test1['repair_postal_code'] = car_insurance_data_test1['third_party_3_postal_code'].apply(str)

print('The shape of the dataset dummy encoded equals ' + str(car_insurance_data_train1.shape))


mean_fraud_amounts = cost[np.nonzero(cost)].mean()
binaire_cost = cost.copy()




binaire_cost[binaire_cost <= mean_fraud_amounts] = 0
binaire_cost[binaire_cost > mean_fraud_amounts] = 1
binaire_cost = np.multiply(np.array(binaire_cost), np.array(fraud_indicators))


column_names_1= datahandler.DataHandler.\
    feature_selector(car_insurance_data_train1, car_insurance_data_test1, None, fraud_indicators)


column_names_2= datahandler.DataHandler.\
    feature_selector(car_insurance_data_train1, car_insurance_data_test1, None, binaire_cost)

final_list = list(set(column_names_1) | set(column_names_2))

car_insurance_data_train1 = car_insurance_data_train1[final_list]
car_insurance_data_test1 = car_insurance_data_test1[final_list]

column_names_3= datahandler.DataHandler.\
    feature_selector_3(car_insurance_data_train1, car_insurance_data_test1, None, fraud_indicators)

column_names_4= datahandler.DataHandler.\
    feature_selector_3(car_insurance_data_train1, car_insurance_data_test1, None, binaire_cost)

final_list = list(set(column_names_3) | set(column_names_4))

car_insurance_data_train1 = car_insurance_data_train1[final_list]
car_insurance_data_test1 = car_insurance_data_test1[final_list]

#
# column_names_5= datahandler.DataHandler.\
#     feature_selector_4(car_insurance_data_train1, car_insurance_data_test1, None, fraud_indicators)
#
# column_names_6= datahandler.DataHandler.\
#     feature_selector_4(car_insurance_data_train1, car_insurance_data_test1, None, binaire_cost)
#
# final_list = list(set(column_names_5) | set(column_names_6))
#
# car_insurance_data_train1 = car_insurance_data_train1[final_list]
# car_insurance_data_test1 = car_insurance_data_test1[final_list]

datahandler.DataHandler.information_value(car_insurance_data_train1, None,
                                          fraud_indicators, 'Fraud')

datahandler.DataHandler.information_value(car_insurance_data_train1, None,
                                          binaire_cost, 'Claim_Amount')



# datahandler.DataHandler.information_value(car_insurance_data_train1, None,
#                                           binaire_cost, 'Claim_Amount')

data_train_no_woe = car_insurance_data_train1.copy()

car_insurance_data_cs_train,car_insurance_data_cs_test,column_names = datahandler.DataHandler.\
    feature_selector_transformer(car_insurance_data_train1, car_insurance_data_test1, None, fraud_indicators)



#all values which are not numeric are replaced with suitable woe value

#woe_ratio = np.log((55463-308)/308)

def isnumber(x):
    return isinstance(x, (int, float))

car_insurance_data_cs_test = car_insurance_data_cs_test[car_insurance_data_cs_test.applymap(isnumber)]
car_insurance_data_cs_test = car_insurance_data_cs_test.apply(lambda x: pd.to_numeric(x, errors='coerce') )

#car_insurance_data_cs_test = car_insurance_data_cs_test.apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(woe_ratio) )


"""
Confirm the data which we are going to work with
"""

data_train = car_insurance_data_cs_train.copy()
data_test = car_insurance_data_cs_test.copy()

print('The shape of the dataset which we are going to work with ' + str(data_train.shape))



covariates = data_train.copy()

fixed_cost = mean_fraud_amounts


"""
Learning
"""

methods = ['CS_XGBoost']


optimal_hyperparameter_finder = 'n'


"""
Cross-validations
"""

# if optimal_hyperparameter_finder == 'Yes':
#     opt_par_dict = Assignment_1.testing_and_plotting.hyperpar_finder(methods,cost,
#                                             data_train, fraud_indicators, fold = 2, repeats = 1)
#     print(opt_par_dict)

if optimal_hyperparameter_finder == 'Yes':
    opt_par_dict = Assignment_1.testing_and_plotting.hyperpar_finder(methods,data_train_no_woe,fraud_indicators,
                                                                     cost,fixed_cost, fold = 2, repeats = 3)
    print(opt_par_dict)

#best results until now

opt_par_dict = {'opt_max_depth': 5, 'opt_n_estimators': 1000, 'opt_lamb_xg': 0.1, 'opt_dropout': 0.3, 'opt_learning_rate': 0.01,
               'cs_opt_max_depth': 5, 'cs_opt_n_estimators': 1000, 'cs_opt_lambd_xg': 0.1, 'cs_opt_dropout': 0.3, 'cs_opt_learning_rate': 0.01}

# 46000 bij de cs

#opt_par_dict = {'opt_max_depth': 5, 'opt_n_estimators': 1000, 'opt_lamb_xg': 5, 'opt_dropout': 0.3, 'opt_learning_rate': 0.01,
#               'cs_opt_max_depth': 5, 'cs_opt_n_estimators': 1000, 'cs_opt_lambd_xg': 5, 'cs_opt_dropout': 0.3, 'cs_opt_learning_rate': 0.01}


# 46000 bij de normale

#opt_par_dict = {'opt_max_depth': 5, 'opt_n_estimators': 1000, 'opt_lamb_xg': 0.1, 'opt_dropout': 0.3, 'opt_learning_rate': 0.01,
#               'cs_opt_max_depth': 5, 'cs_opt_n_estimators': 1000, 'cs_opt_lambd_xg': 5, 'cs_opt_dropout': 0.6, 'cs_opt_learning_rate': 0.01}

"""
Get test file probs
"""


fraud_indicators_train = fraud_indicators.values


xgboost = xgbst.Xgbst(n_estimators =opt_par_dict.get("opt_n_estimators"),
                      max_depth = opt_par_dict.get("opt_max_depth"),
                      lambd = opt_par_dict.get("opt_lamb_xg"),
                      dropout= opt_par_dict.get("opt_dropout"), learning_rate = opt_par_dict.get("opt_learning_rate"))
xgbst_train, time = xgboost.fitting(data_train,fraud_indicators_train)
xgpred = xgboost.predict_proba(xgbst_train,data_test)


dataframe_xgbst_output  = pd.DataFrame(xgpred,index=test_index)

Assignment_1.testing_and_plotting.feature_imp_impurity(xgbst_train, 'boost')
Assignment_1.testing_and_plotting.feature_imp_shap(xgbst_train, data_train,'boost')


cs_xgboost = cs_xgboost.CSXgboost(n_estimators =opt_par_dict.get("cs_opt_max_depth"),
                                  max_depth = opt_par_dict.get("cs_opt_n_estimators"),
                                  lambd = opt_par_dict.get("cs_opt_lambd_xg"),
                                  dropout= opt_par_dict.get("cs_opt_dropout"),
                                  learning_rate = opt_par_dict.get("opt_learning_rate"),
                                  fixed_cost=fixed_cost)
cs_xgboost_train, time = cs_xgboost.fitting(data_train,fraud_indicators_train,cost)
cs_xgpred = cs_xgboost.predict_proba(cs_xgboost_train,data_test)

dataframe_cs_xgbst_output  = pd.DataFrame(cs_xgpred,index=test_index)

Assignment_1.testing_and_plotting.feature_imp_impurity(cs_xgboost_train, 'cs_boost')
Assignment_1.testing_and_plotting.feature_imp_shap(cs_xgboost_train, data_train,'cs_boost')


base_path = Path(__file__).parent

data_path = (base_path / "../assign_1/Assignment_1/output.csv").resolve()
dataframe_xgbst_output.to_csv(data_path, index=True, header=None)

data_path = (base_path / "../assign_1/Assignment_1/cs_output.csv").resolve()
dataframe_cs_xgbst_output.to_csv(data_path, index=True, header=None)
