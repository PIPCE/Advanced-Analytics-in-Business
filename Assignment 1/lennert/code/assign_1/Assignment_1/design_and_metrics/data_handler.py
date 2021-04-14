import pandas as pd
import category_encoders as ce
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xverse.transformer import WOE
from xverse.ensemble import VotingSelector
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

class DataHandler:

    @staticmethod
    def create_dummies(data_na, columns):


        # for col in categorical_covariates:
        #     if col in data_na.columns:
        #             data_na = pd.get_dummies(data=data_na, columns=[col], dummy_na = False)

        na_string = "_nan"
        for col in columns:
             if col in data_na.columns:
                 column = col + na_string
                 data_na[column] =  data_na[col].isnull().astype(int)

        return data_na

    @staticmethod
    def impute_median(data_na, count_and_continuous_covariates):

        for col in count_and_continuous_covariates:
            if col in data_na.columns:
                data_na[col].fillna((data_na[col].median()), inplace = True)
        return data_na

    @staticmethod
    def impute_mode(data_na, categorical_values):

        for col in categorical_values:
            if col in data_na.columns:
                data_na[col].fillna((data_na[col].mode()), inplace=True)
        return data_na

    @staticmethod
    def remove_na(data_na, column):

        if column in data_na.columns:
            data = data_na.dropna(subset=[column])
        print('The shape of the dataset not including empty values equals ' + str(data.shape))
        return data

    @staticmethod
    def categorical_binner(row, column_name, lists, bin_list):
        for i, list in enumerate(lists):
            if row[column_name] in list:
                return bin_list[i]

    @staticmethod
    def information_value(data, exclude_columns, column_responses, string):
        clf = WOE(treat_missing='separate', exclude_features=exclude_columns)
        clf.fit(data, column_responses)
        test = clf.woe_bins
        info_value = clf.iv_df
        info_value = info_value.nlargest(20, 'Information_Value')
        ax = sns.barplot(x=info_value['Variable_Name'], y=info_value['Information_Value'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",fontsize=7)
        plt.title('Feature Importance')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(string+"_Information_Value.png")
        plt.close()

    @staticmethod
    def feature_selector_transformer(data_train, data_test, exclude_columns, column_responses):

        #https://towardsdatascience.com/introducing-xverse-a-python-package-for-feature-selection-and-transformation-17193cdcd067
        #https: // github.com / Sundar0989 / XuniVerse
        #https: // github.com / Sundar0989 / XuniVerse / blob / master / xverse / transformer / _woe.py

        # All categorical features in the dataset will be used for woee encoding
        # continuous NANs are imputed with their mean


        clf = VotingSelector(selection_techniques=['WOE','RF','ETC'],exclude_features=exclude_columns,
                             handle_category='woe',numerical_missing_values= 'median',minimum_votes=3)

        clf.fit(data_train, column_responses)

        transformed_data_train = clf.transform(data_train)
        transformed_data_test = clf.transform(data_test)
        column_names = transformed_data_train.columns.values

        return transformed_data_train,transformed_data_test,column_names

    @staticmethod
    def feature_selector(data_train, data_test, exclude_columns, column_responses):
        clf = VotingSelector(selection_techniques=['WOE'], exclude_features=exclude_columns,
                             handle_category='woe', numerical_missing_values='median', minimum_votes=1)
        clf.fit(data_train, column_responses)
        transformed_data_train = clf.transform(data_train)
        column_names = transformed_data_train.columns.values
        return list(column_names)

    @staticmethod
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

    @staticmethod
    def deleter(data, columns):
            data = data.drop(columns, axis = 1)
            return  data

    @staticmethod
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
