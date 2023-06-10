'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-06-09 16:09:26
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-06-09 16:32:35
 # @ Description: The *churn_library.py* is
 #  a library of functions to find customers who are likely to churn.
 '''

# import libraries
import logging
import warnings
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

warnings.filterwarnings("ignore")
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(
            "./logs/churn_library_logs.log",
            mode="w"),
        stream_handler],
    format="%(name)s - %(levelname)s - %(message)s",
)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
                    pth: a path to the csv
    output:
                    dataframe: pandas dataframe
    '''
    try:
        dataframe = pd.read_csv(pth)
    except FileNotFoundError:
        logging.error("File csv not found")
        return None
    except pd.errors.ParserError:
        logging.error("Parse file errors")
        return None
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    logging.info("Perform EDA")
    save_dir = "./images/eda"
    # logging.info(dataframe.isnull().sum())
    # logging.info(dataframe.describe())
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    fig = plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    fig.savefig(f'{save_dir}/churn_distribution.png')
    dataframe['Customer_Age'].hist()
    fig.savefig(f'{save_dir}/customer_age_distribution.png')
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    fig.savefig(f'{save_dir}/marital_status_distribution.png')
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    fig.savefig(f'{save_dir}/total_trans_distribution.png')
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig.savefig(f'{save_dir}/heatmap.png')


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
            for naming variables or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    '''
    for cat_col in category_lst:
        cat_lst = []
        cat_groups = dataframe.groupby(cat_col).mean()['Churn']

        for val in dataframe[cat_col]:
            cat_lst.append(cat_groups.loc[val])

        if response:
            dataframe[cat_col + '_' + response] = cat_lst
        else:
            dataframe[cat_col] = cat_lst
    return dataframe


def perform_feature_engineering(dataframe, response):
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    logging.info("Perform feature engineering")
    y_target = dataframe['Churn']
    x_dataframe = pd.DataFrame()
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    dataframe = encoder_helper(dataframe, cat_columns, response=response)
    keep_cols = [
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

    x_dataframe[keep_cols] = dataframe[keep_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x_dataframe, y_target, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions from logistic regression and random forest
            y_test_preds: test predictions from logistic regression and random forest
    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))

    y_train_preds_lr, y_train_preds_rf = y_train_preds
    y_test_preds_lr, y_test_preds_rf = y_test_preds

    values = [
        ('Random Forest',
         y_train,
         y_train_preds_rf,
         y_test,
         y_test_preds_rf,
         'random_forest_classification_report.png'),
        ('Logistic Regression',
         y_train,
         y_train_preds_lr,
         y_test,
         y_test_preds_lr,
         'logistic_regression_classification_report.png')]
    for value in values:
        plt.text(0.01, 1.25, str(f'{value[0]} Train'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(value[1], value[2])), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(f'{value[0]} Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(value[3], value[4])), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig(f'./images/results/{value[5]}')


def feature_importance_plot(model, x_dataframe, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_dataframe: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_dataframe.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_dataframe.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_dataframe.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    logging.info("Perform train model")
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    logging.info("Perform prediction")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    logging.info("Perform evaluate model")
    classification_report_image(
        y_train,
        y_test,
        (y_train_preds_lr,
         y_train_preds_rf),
        (y_test_preds_lr,
         y_test_preds_rf))

    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    plt.savefig("./images/results/logistic_resuts.png")

    plt.figure(figsize=(15, 8))
    plt_ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=plt_ax,
        alpha=0.8)
    lrc_plot.plot(ax=plt_ax, alpha=0.8)
    plt.savefig("./images/results/roc_curve_results.png")

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap.summary_plot(explainer.shap_values(x_test), x_test, plot_type="bar")
    plt.savefig("./images/results/shap_summary.png")

    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_dataframe=x_train,
        output_pth="./images/results/feature_importances.png")

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    BANK_DF = import_data("data/bank_data.csv")

    perform_eda(BANK_DF)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        BANK_DF, response="Churn")
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
