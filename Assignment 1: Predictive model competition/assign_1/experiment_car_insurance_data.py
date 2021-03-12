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



car_insurance_data = datahandler.DataHandler.deleter(car_insurance_data,
                       ['claim_vehicle_id', 'policy_holder_id',
                                          'driver_id',
                                          'driver_vehicle_id',
                                          'third_party_1_id',
                                          'third_party_1_vehicle_id',
                                          'third_party_1_expert_id',
                                          'third_party_2_id',
                                          'third_party_2_vehicle_id',
                                          'third_party_2_expert_id',
                                          'third_party_3_id',
                                          'third_party_3_vehicle_id',
                                          'third_party_3_expert_id',
                                          'repair_id',
                        'driver_expert_id', 'policy_holder_expert_id'])

#person time related columns

car_insurance_data = datahandler.\
    DataHandler.deleter(car_insurance_data,
                 ['third_party_1_year_birth',
                                          'third_party_2_year_birth',
                                          'third_party_3_year_birth',
                                          'repair_year_birth'])

#'repair_year_birth'

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

#claim_time_occured
#claim_vehicle_date_inuse does not work for some reason


#postal code and country related columns

car_insurance_data = datahandler.DataHandler.deleter(car_insurance_data,
                      [  'repair_postal_code','repair_country'])

car_insurance_data_train1 = car_insurance_data.iloc[:training_instances]
car_insurance_data_test1 = car_insurance_data.iloc[training_instances:]

#party columns


#car_insurance_data_train, car_insurance_data_test = datahandler.DataHandler.deleter(car_insurance_data_train, car_insurance_data_test,
#                              [])

#extra columns appear when one hot encoding the train and test set
#This will be solved by these lines

"""
Weight of evidence encoding

claim_vehicle_brand
policy_coverage_type
"""

woee1, woee1_test= datahandler.DataHandler.\
    weight_of_evidence_encoding(car_insurance_data_train1,
                car_insurance_data_test1, 'claim_vehicle_brand', fraud_indicators)

woee2, woee2_test = datahandler.DataHandler.\
    weight_of_evidence_encoding(woee1, woee1_test,
                            'policy_coverage_type', fraud_indicators)

woee3, woee3_test = datahandler.DataHandler.\
    weight_of_evidence_encoding(woee2, woee2_test,
                            'claim_postal_code', fraud_indicators)

woee4, woee4_test = datahandler.DataHandler.\
    weight_of_evidence_encoding(woee3, woee3_test,
                            'policy_holder_postal_code', fraud_indicators)

woee5, woee5_test = datahandler.DataHandler.\
    weight_of_evidence_encoding(woee4, woee4_test,
                            'driver_postal_code', fraud_indicators)

woee6, woee6_test = datahandler.DataHandler.\
    weight_of_evidence_encoding(woee5, woee5_test,
                            'third_party_1_postal_code', fraud_indicators)

woee7, woee7_test = datahandler.DataHandler.\
    weight_of_evidence_encoding(woee6, woee6_test,
                            'third_party_2_postal_code', fraud_indicators)

woee8, woee8_test = datahandler.DataHandler.\
    weight_of_evidence_encoding(woee7, woee7_test,
                            'third_party_3_postal_code', fraud_indicators)


"""
One hot encoding categorical
"""

categorical_covariates = ['claim_cause','claim_liable','claim_police',
                            'claim_alcohol','third_party_2_injured','third_party_3_injured',
                          'claim_language', 'claim_vehicle_type',
                          'policy_holder_country',  'driver_country', 'third_party_1_country',
                          'claim_vehicle_fuel_type','policy_holder_form','driver_form',
                          'driver_injured', 'third_party_1_injured', 'third_party_1_vehicle_type',
                          'third_party_1_form','third_party_2_vehicle_type','third_party_2_form',
                                              'third_party_3_form','third_party_3_vehicle_type',
                           'third_party_2_country',
                                          'third_party_3_country',
                          'repair_form', 'repair_sla']


dummy_creater = woee8.append(woee8_test)

car_insurance_data1 = datahandler.DataHandler.create_dummies(dummy_creater.copy(), categorical_covariates)
car_insurance_data_cs_dummy = car_insurance_data1.iloc[:training_instances]
car_insurance_data_cs_test = car_insurance_data1.iloc[training_instances:]

# car_insurance_data_cs_dummy = datahandler.DataHandler.create_dummies(woee8.copy(), categorical_covariates)
#
# car_insurance_data_cs_test= datahandler.DataHandler.\
#     create_dummies(woee8_test.copy(), categorical_covariates)


"""
Confirm the data which we are going to work with
"""

data_train = car_insurance_data_cs_dummy.copy()
data_test = car_insurance_data_cs_test.copy()

print('The shape of the dataset which we are going to work with ' + str(data_train.shape))


"""
Learning
"""

methods = ['XGBoost', 'CS_XGBoost']

optimal_hyperparameter_finder = 'Yes'


"""
Cross-validations
"""

if optimal_hyperparameter_finder == 'Yes':
    opt_par_dict = Assignment_1.testing_and_plotting.hyperpar_finder(methods,
                                                                     data_train, fraud_indicators, fold = 2, repeats = 1)
    print(opt_par_dict)

opt_par_dict = {'opt_max_depth': 5, 'opt_n_estimators': 500, 'opt_lamb_xg': 0.01, 'opt_dropout':0.75,"opt_learning_rate:": 0.01,
                'cs_opt_max_depth': 5, 'cs_opt_n_estimators': 500, 'cs_opt_lambd_xg': 0.01, 'cs_opt_dropout':0.75, 'cs_opt_learning_rate':0.01}


"""
Get test file probs
"""


fraud_indicators_train = fraud_indicators.values



xgboost = xgbst.Xgbst(n_estimators =opt_par_dict.get("opt_n_estimators"),
                      max_depth = opt_par_dict.get("opt_max_depth"),
                      lambd = opt_par_dict.get("opt_lamb_xg"),
                      dropout= opt_par_dict.get("opt_dropout"), learning_rate = 0.01)
xgbst_train, time = xgboost.fitting(data_train,fraud_indicators_train)
xgpred = xgboost.predict_proba(xgbst_train,data_test)


dataframe_xgbst_output  = pd.DataFrame(xgpred,index=test_index)

Assignment_1.testing_and_plotting.feature_imp(xgbst_train)

cs_xgboost = cs_xgboost.CSXgboost(n_estimators =opt_par_dict.get("cs_opt_max_depth"),
                                  max_depth = opt_par_dict.get("cs_opt_n_estimators"),
                                  lambd = opt_par_dict.get("cs_opt_lambd_xg"),
                                  dropout= opt_par_dict.get("cs_opt_dropout"),
                                  learning_rate = 0.01)
cs_xgboost_train, time = cs_xgboost.fitting(data_train,fraud_indicators_train)
cs_xgpred = cs_xgboost.predict_proba(cs_xgboost_train,data_test)

dataframe_cs_xgbst_output  = pd.DataFrame(cs_xgpred,index=test_index)

Assignment_1.testing_and_plotting.feature_imp(cs_xgboost_train)


base_path = Path(__file__).parent

data_path = (base_path / "../Cost_Sensitive_Learning/Assignment_1/output.csv").resolve()
dataframe_xgbst_output.to_csv(data_path, index=True, header=None)

data_path = (base_path / "../Cost_Sensitive_Learning/Assignment_1/cs_output.csv").resolve()
dataframe_cs_xgbst_output.to_csv(data_path, index=True, header=None)
