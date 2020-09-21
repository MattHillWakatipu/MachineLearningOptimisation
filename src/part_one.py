# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Init settings
seed = 309
random.seed(seed)
np.random.seed(seed)


# load data
def load_part_one():
    df = pd.read_csv("../data/Part 1 - regression/diamonds.csv")
    return df


if __name__ == '__main__':
    # Settings
    metric_type = "MSE"  # MSE, RMSE, MAE, R2
    optimizer_type = "BGD"  # PSO, BGD

    # Step 1: Load Data
    data = load_part_one()

    # Step 2: Initial Data Analysis
    df = pd.DataFrame(data)
    print(df.head())
    print("Shape:", df.shape)
    print("Missing data?:", df.isnull().values.any())

    # Step 3: Preprocess the data
    train_data, test_data, = train_test_split(df, test_size=0.3, random_state=seed)

    # Step 4: Exploratory Data analysis
    with pd.option_context('display.max_columns', 11):
        print("\nDescribe:")
        print(train_data.describe(include='all'))
        print("\nCorrelation:")
        print(train_data.corr())

    # TODO work out how to make plots
    # train_data.plot.scatter(x='price', y='carat')

    # FIXME wtf is this?
    # Make a copy of test data
    test_data_full = test_data.copy()
    # Drop the ground truth
    test_data = test_data.drop(["price"], axis=1)
    # Get test labels
    test_labels = test_data_full["price"]
    print(test_data.head())

    # FIXME NaN fields when Standardize the data
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    print(train_data.head())

    # Step 5: Build models using training data
    # theta = np.array([0.0, 0.0])  # Initialize model parameter

    # start_time = datetime.datetime.now()  # Track learning starting time
    # thetas, losses = learn(train_labels.values, train_data.values, theta, max_iters, alpha, optimizer_type, metric_type)

    # end_time = datetime.datetime.now()  # Track learning ending time
    # exection_time = (end_time - start_time).total_seconds()  # Track execution time

    # Step 4: Results presentation
    # print("Learn: execution time={t:.3f} seconds".format(t = exection_time))

    # Build baseline model
    # print("R2:", -compute_loss(test_labels.values, test_data.values, thetas[-1], "R2"))  # R2 should be maximize
    # print("MSE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "MSE"))
    # print("RMSE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "RMSE"))
    # print("MAE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "MAE"))
