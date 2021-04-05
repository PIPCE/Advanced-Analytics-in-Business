import pandas as pd
import importlib.resources
from pathlib import Path


def import_car_insurance_data():


    base_path = Path(__file__).parent
    # file_path_train = (base_path / "../car_insurance_data/train.xlsx").resolve()
    # data_train = pd.read_excel(file_path_train, index_col = 0)
    #
    # file_path_test = (base_path / "../car_insurance_data/test.xlsx").resolve()
    # data_test = pd.read_excel(file_path_test, index_col = 0)
    #
    # data_path_train = (base_path / "../car_insurance_data/data_train.csv").resolve()
    # data_path_test = (base_path / "../car_insurance_data/data_test.csv").resolve()
    #
    # data_train.to_csv(data_path_train, index=True)
    # data_test.to_csv(data_path_test, index=True)


    file_path_train = (base_path / "../car_insurance_data/data_train.csv").resolve()
    data_train = pd.read_csv(file_path_train, index_col =0)

    file_path_test = (base_path / "../car_insurance_data/data_test.csv").resolve()
    data_test = pd.read_csv(file_path_test, index_col = 0)

    return data_train, data_test
