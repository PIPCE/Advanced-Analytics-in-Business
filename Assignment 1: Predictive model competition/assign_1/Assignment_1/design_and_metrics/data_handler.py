import pandas as pd
import category_encoders as ce
import numpy as np

class DataHandler:

    def create_dummies(data_na, categorical_covariates):


        for col in categorical_covariates:
            if col in data_na.columns:
                    data_na = pd.get_dummies(data=data_na, columns=[col], dummy_na = False)
        return data_na


    def impute_median(data_na, count_and_continuous_covariates):

        for col in count_and_continuous_covariates:
            if col in data_na.columns:
                data_na[col].fillna((data_na[col].median()), inplace = True)
        return data_na

    def impute_mode(data_na, categorical_values):

        for col in categorical_values:
            if col in data_na.columns:
                data_na[col].fillna((data_na[col].mode()), inplace=True)
        return data_na

    def remove_na(data_na, column):

        if column in data_na.columns:
            data = data_na.dropna(subset=[column])
        print('The shape of the dataset not including empty values equals ' + str(data.shape))
        return data


    def categorical_binner(row, column_name, lists, bin_list):
        for i, list in enumerate(lists):
            if row[column_name] in list:
                return bin_list[i]


    def weight_of_evidence_encoding(data_train, data_test, column_categorical, column_responses):

        enc = ce.woe.WOEEncoder(handle_missing='return_nan',handle_unknown='return_nan', randomized=True)
        enc.fit(data_train[column_categorical], column_responses)
        woee = enc.transform(data_train[column_categorical])
        woee_test = enc.transform(data_test[column_categorical])
        data_copy = data_train.copy()
        data_test_copy = data_test.copy()
        data_copy[column_categorical] = woee
        data_test_copy[column_categorical] = woee_test
        return data_copy,data_test_copy

    def deleter(data, columns):
            data = data.drop(columns, axis = 1)
            return  data

    def date_ymd_converter(data, columns):
        data = data.copy()
        for col in columns:
            data[col] = pd.to_datetime(data[col],errors='ignore',format='%Y%m%d')

        return data

    def date_ym_converter(data, columns):
        data= data.copy()
        for col in columns:
            data[col] = pd.to_datetime(data[col],errors='ignore',format='%Y%m')

        return data



    def date_month_differencer(data,column1, column2):

        data = data.copy()

        data[column1+'_'+column2+'_diff'] = data[column1].sub(data[column2], axis=0).dt.days

        return data
