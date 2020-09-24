# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Init settings
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

from utilities.losses import compute_loss

seed = 309
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3


# load data
def load_part_one():
    df = pd.read_csv("../data/Part 1 - regression/diamonds.csv")
    return df


def data_preprocess(data):
    """
    Data preprocess:
        1. Split the entire dataset into train and test
        2. Split outputs and inputs
        3. Standardize train and test
        4. Add intercept dummy for computation convenience
    :param data: the given dataset (format: panda DataFrame)
    :return: train_data       train data contains only inputs
             train_labels     train data contains only labels
             test_data        test data contains only inputs
             test_labels      test data contains only labels
             train_data_full       train data (full) contains both inputs and labels
             test_data_full        test data (full) contains both inputs and labels
    """
    # Categorical conversion
    lb_make = LabelEncoder()
    data['cut'] = lb_make.fit_transform(data['cut'])
    data['color'] = lb_make.fit_transform(data['color'])
    data['clarity'] = lb_make.fit_transform(data['clarity'])

    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size=train_test_split_test_size)

    # Pre-process data (both train and test)
    train_data_full = train_data.copy()
    train_data = train_data.drop(["price"], axis=1)
    train_labels = train_data_full["price"]

    test_data_full = test_data.copy()
    test_data = test_data.drop(["price"], axis=1)
    test_labels = test_data_full["price"]

    # Standardize the inputs
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    # Tricks: add dummy intercept to both train and test
    train_data['intercept_dummy'] = pd.Series(1.0, index=train_data.index)
    test_data['intercept_dummy'] = pd.Series(1.0, index=test_data.index)
    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


def createPieChart(cat_data, category):
    labels = cat_data[category].astype('category').cat.categories.tolist()
    counts = cat_data[category].value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)  # autopct is show the % on plot
    ax1.axis('equal')
    plt.show()


def calculate_errors(model_name, model):
    price_pred = model.predict(test_data)
    mse = mean_squared_error(test_labels, price_pred)
    r2_error = r2_score(test_labels, price_pred)
    mae = mean_absolute_error(test_labels, price_pred)

    print(model_name)
    print("MSE: ", mse)
    print("RMSE: ", np.sqrt(mse))
    print("R2: ", r2_error)
    print("MAE: ", mae)
    print()


if __name__ == '__main__':
    # Settings
    metric_type = "MSE"  # MSE, RMSE, MAE, R2
    optimizer_type = "BGD"  # PSO, BGD

    # Step 1: Load Data
    data = load_part_one()

    # Step 2: Initial Data Analysis
    df = pd.DataFrame(data)
    with pd.option_context('display.max_columns', 11):
        print(df.head())
        print("\nShape:", df.shape)
        print("\nMissing data?:", df.isnull().values.any())
        print("\nData Types: ", df.info())

    # Step 3: Preprocess the data
    train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(data)

    print("\nData Types: ", df.info())

    print("\nTest Data Head: \n", test_data.head())

    # Step 4: Exploratory Data analysis

    # TODO work out how to make plots
    # train_data.plot.scatter(x='price', y='carat')
    #
    # with pd.option_context('display.max_columns', 11):
    #     print("\nDescribe:")
    #     print(train_data.describe(include='all'))
    #     print("\nCorrelation:")
    #     print(train_data.corr())

    # Step 5: Build models & Evaluate on the test set
    linear_regression = LinearRegression()
    linear_regression.fit(train_data, train_labels)
    calculate_errors("Linear Regression", linear_regression)

    knn = KNeighborsRegressor(5)
    knn.fit(train_data, train_labels)
    calculate_errors("K-Neighbours Regression", knn)

    ridge = Ridge()
    ridge.fit(train_data, train_labels)
    calculate_errors("Ridge Regression", ridge)

    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(train_data, train_labels)
    calculate_errors("Decision Tree Regression", decision_tree)

    random_forest = RandomForestRegressor()
    random_forest.fit(train_data, train_labels)
    calculate_errors("Random Forest Regression", random_forest)

    boosting = GradientBoostingRegressor()
    boosting.fit(train_data, train_labels)
    calculate_errors("Gradient Boosting Regression", boosting)

    sgd = linear_model.SGDRegressor()
    sgd.fit(train_data, train_labels)
    calculate_errors("SGD Regression", sgd)

    svr = SVR()
    svr.fit(train_data, train_labels)
    calculate_errors("Support Vector Regression", svr)

    linear_svr = LinearSVR()
    linear_svr.fit(train_data, train_labels)
    calculate_errors("Linear Support Vector Regression", linear_svr)

    mlp = MLPRegressor(max_iter=1000)
    mlp.fit(train_data, train_labels)
    calculate_errors("Multi-Layer Perceptron Regression", mlp)
