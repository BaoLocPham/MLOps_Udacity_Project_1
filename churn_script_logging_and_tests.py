'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-06-09 16:09:22
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-06-10 10:11:30
 # @ Description: This file is for testing churn library
 '''

import os
import logging
import warnings
import churn_library as clb
# import pytest
warnings.filterwarnings("ignore")

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# @pytest.fixture(scope="module")
# def response_str():
#     """
#     pytest Fixture response_str
#     """
#     return "Churn"


# @pytest.fixture(scope="module")
# def path():
#     """
#     pytest Fixture path
#     """
#     return "./data/bank_data.csv"


# @pytest.fixture(scope="module")
# def data(path):
#     """
#     pytest Fixture path
#     """
#     return clb.import_data(path)


# @pytest.fixture(scope="module")
# def train_test_data(data, response_str):
#     """
#     pytest Fixture train_test_data
#     """
#     return clb.perform_feature_engineering(data, response_str)


# @pytest.fixture(scope="module")
# def eda_plot_paths():
#     """
#     pytest Fixture eda_plot_paths
#     """
#     return [
#         r"images/eda/churn_distribution.png",
#         r"images/eda/customer_age_distribution.png",
#         r"images/eda/heatmap.png",
#         r"images/eda/marital_status_distribution.png",
#         r"images/eda/total_trans__distribution.png",
#     ]


# @pytest.fixture(scope="module")
# def train_plot_paths():
#     """
#     pytest Fixture train_plot_paths
#     """
#     return [
#         r"images/results/feature_importances.png",
#         r"images/results/logistic_regression_classification_report.png",
#         r"images/results/logistic_resuts.png",
#         r"images/results/random_forest_classification_report.png",
#         r"images/results/roc_curve_results.png",
#         r"images/results/shap_summary.png",
#     ]


# @pytest.fixture(scope="module")
# def train_model_paths():
#     """
#     pytest Fixture train_model_paths
#     """
#     return [
#         r"models/logistic_model.pkl",
#         r"models/rfc_model.pkl",
#     ]


# @pytest.fixture(scope="module")
# def category_list():
#     """
#     pytest Fixture category_list
#     """
#     return [
#         'Gender',
#         'Education_Level',
#         'Marital_Status',
#         'Income_Category',
#         'Card_Category'
#     ]


# @pytest.fixture(scope="module")
# def keep_cols_list():
#     """
#     pytest Fixture keep_cols_list
#     """
#     return [
#         'Customer_Age',
#         'Dependent_count',
#         'Months_on_book',
#         'Total_Relationship_Count',
#         'Months_Inactive_12_mon',
#         'Contacts_Count_12_mon',
#         'Credit_Limit',
#         'Total_Revolving_Bal',
#         'Avg_Open_To_Buy',
#         'Total_Amt_Chng_Q4_Q1',
#         'Total_Trans_Amt',
#         'Total_Trans_Ct',
#         'Total_Ct_Chng_Q4_Q1',
#         'Avg_Utilization_Ratio',
#         'Gender_Churn',
#         'Education_Level_Churn',
#         'Marital_Status_Churn',
#         'Income_Category_Churn',
#         'Card_Category_Churn']


def test_import_data(path):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	logging.info("Testing import_data")
	try:
		dataframe = clb.import_data(path)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert len(dataframe) > 0
		assert len(dataframe.columns) > 0
	except AssertionError as err:
		logging.error(
			"Testing import_data: The file doesn't appear to have rows and columns")
		raise err
	logging.info("SUCCES test import_data")

def test_perform_eda(data, eda_plot_paths):
	'''
	test perform eda function
	'''
	logging.info("Testing perform_eda")
	try:
		clb.perform_eda(data)
		for plot_path in eda_plot_paths:
			plot_path = os.path.join(os.getcwd(), plot_path)
			assert os.path.exists(plot_path)
	except AssertionError as err:
		logging.error("Test perform_eda: file not found")
		raise err
	logging.info("SUCCES test perform_eda")

def test_encoder_helper(data, category_list, response_str):
	'''
	test encoder helper
	'''
	logging.info("Testing encoder_helper")
	try:
		encoded_data = clb.encoder_helper(data, category_list, response_str)
		assert len(encoded_data.columns) - \
			len(category_list) != len(data.columns)
	except AssertionError as err:
		logging.error(
			"Test encoder_helper: encoded_data doesn't have enough encoded colums")
		raise err
	logging.info("SUCCES test encoder_helper")

def test_perform_feature_engineering(data, response_str, keep_cols_list):
	'''
	test perform_feature_engineering
	'''
	logging.info("Testing perform_feature_engineering")
	try:
		x_train, x_test, y_train, y_test = clb.perform_feature_engineering(
			data, response_str)
		assert x_train.shape[0] > 0
		assert x_test.shape[0] > 0
		assert y_train.shape[0] > 0
		assert y_test.shape[0] > 0

		assert (x_train.columns == keep_cols_list).all
		assert (x_test.columns == keep_cols_list).all
	except AssertionError as err:
		logging.error(
			"Test perform_feature_engineering: Error in shape, column assertion")
		raise err
	logging.info("SUCCES test perform_feature_engineering")


def test_train_models(train_test_data, train_plot_paths, train_model_paths):
	'''
	test train_models
	'''
	logging.info("Testing train_models")
	try:
		x_train, x_test, y_train, y_test = train_test_data
		clb.train_models(x_train, x_test, y_train, y_test)

		for plot_path in train_plot_paths:
			assert os.path.exists(plot_path)

		for model_path in train_model_paths:
			assert os.path.exists(model_path)

	except AssertionError as err:
		logging.error("Test encoder_helper: Eror in file existance assertion")
		raise err
	logging.info("SUCCES test train_models")

if __name__ == "__main__":
    PATH_DATA = "./data/bank_data.csv"
    test_import_data(PATH_DATA)
    EDA_PLOT_PATHS = [
        r"images/eda/churn_distribution.png",
        r"images/eda/customer_age_distribution.png",
        r"images/eda/heatmap.png",
        r"images/eda/marital_status_distribution.png",
        r"images/eda/total_trans_distribution.png",
    ]
    DATA = clb.import_data(PATH_DATA)
    test_perform_eda(DATA, EDA_PLOT_PATHS)
    CATEGORY_LIST = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    KEEP_COLS_LIST = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    RESPONSE_STR = "Churn"
    TRAIN_PLOT_PATHS = [
        r"images/results/feature_importances.png",
        r"images/results/logistic_regression_classification_report.png",
        r"images/results/logistic_resuts.png",
        r"images/results/random_forest_classification_report.png",
        r"images/results/roc_curve_results.png",
        r"images/results/shap_summary.png",
    ]
    TRAIN_MODEL_PATHS = [
        r"models/logistic_model.pkl",
        r"models/rfc_model.pkl",
    ]
    test_encoder_helper(DATA, CATEGORY_LIST, RESPONSE_STR)
    test_perform_feature_engineering(DATA, RESPONSE_STR, KEEP_COLS_LIST)
    TRAIN_TEST_DATA = clb.perform_feature_engineering(DATA, RESPONSE_STR)
    test_train_models(TRAIN_TEST_DATA, TRAIN_PLOT_PATHS, TRAIN_MODEL_PATHS)
