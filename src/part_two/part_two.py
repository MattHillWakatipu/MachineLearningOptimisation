# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

from sklearn import linear_model, naive_bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score

# Init settings
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR, LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from utilities.losses import compute_loss

seed = 309
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3


# load data
def load_part_two():
    df_data = pd.read_csv("../../data/Part 2 - classification/adult.data")
    df_test = pd.read_csv("../../data/Part 2 - classification/adult.test")
    return df_data, df_test

def data_preprocess(data, test):
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
             test_data_full       test data (full) contains both inputs and labels
    """
    # Categorical conversion
    lb_make = LabelEncoder()
    data['workclass'] = lb_make.fit_transform(data['workclass'])
    data['education'] = lb_make.fit_transform(data['education'])
    data['marital-status'] = lb_make.fit_transform(data['marital-status'])
    data['occupation'] = lb_make.fit_transform(data['occupation'])
    data['relationship'] = lb_make.fit_transform(data['relationship'])
    data['race'] = lb_make.fit_transform(data['race'])
    data['sex'] = lb_make.fit_transform(data['sex'])
    data['hours-per-week'] = lb_make.fit_transform(data['hours-per-week'])
    data['native-country'] = lb_make.fit_transform(data['native-country'])
    data['class'] = lb_make.fit_transform(data['class'])

    test['workclass'] = lb_make.fit_transform(test['workclass'])
    test['education'] = lb_make.fit_transform(test['education'])
    test['marital-status'] = lb_make.fit_transform(test['marital-status'])
    test['occupation'] = lb_make.fit_transform(test['occupation'])
    test['relationship'] = lb_make.fit_transform(test['relationship'])
    test['race'] = lb_make.fit_transform(test['race'])
    test['sex'] = lb_make.fit_transform(test['sex'])
    test['hours-per-week'] = lb_make.fit_transform(test['hours-per-week'])
    test['native-country'] = lb_make.fit_transform(test['native-country'])
    test['class'] = lb_make.fit_transform(test['class'])

    # Split the data into train and test
    train_data = data
    test_data = test

    print(train_data.corr())

    # Pre-process data (both train and test)
    train_data_full = train_data.copy()
    train_data = train_data.drop(["class"], axis=1)
    train_labels = train_data_full["class"]

    test_data_full = test_data.copy()
    test_data = test_data.drop(["class"], axis=1)
    test_labels = test_data_full["class"]

    train_data = train_data.drop(["workclass"], axis=1)
    test_data = test_data.drop(["workclass"], axis=1)
    train_data = train_data.drop(["fnlwgt"], axis=1)
    test_data = test_data.drop(["fnlwgt"], axis=1)
    train_data = train_data.drop(["education"], axis=1)
    test_data = test_data.drop(["education"], axis=1)
    train_data = train_data.drop(["occupation"], axis=1)
    test_data = test_data.drop(["occupation"], axis=1)
    train_data = train_data.drop(["race"], axis=1)
    test_data = test_data.drop(["race"], axis=1)
    train_data = train_data.drop(["native-country"], axis=1)
    test_data = test_data.drop(["native-country"], axis=1)

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
    class_pred = model.predict(test_data)
    accuracy = accuracy_score(test_labels, class_pred)
    precision = precision_score(test_labels, class_pred)
    recall = recall_score(test_labels, class_pred)
    f1 = f1_score(test_labels, class_pred)
    auc = roc_auc_score(test_labels, class_pred)

    print(model_name)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1)
    print("AUC: ", auc)
    print()


if __name__ == '__main__':
    # Settings
    metric_type = "MSE"  # MSE, RMSE, MAE, R2
    optimizer_type = "BGD"  # PSO, BGD

    # Step 1: Load Data
    data, test = load_part_two()

    # Step 2: Initial Data Analysis
    df_data = pd.DataFrame(data)
    with pd.option_context('display.max_columns', 20):
        print(df_data.head())
        print("\nShape:", df_data.shape)
        print("\nMissing data?:", df_data.isnull().values.any())
        print("\nData Types: ", df_data.info())
        print("\nDescribe:")
        print(df_data.describe(include='all'))
        print(df_data.corr())

    # Step 3: Preprocess the data
    train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(data, test)

    print("\nData Types: ", df_data.info())

    with pd.option_context('display.max_columns', 20):
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
    knn = KNeighborsClassifier(5)
    knn.fit(train_data, train_labels)
    calculate_errors("K-Neighbours", knn)

    bayes = naive_bayes.BernoulliNB()
    bayes.fit(train_data, train_labels)
    calculate_errors("Naive Bayes", bayes)

    svm = LinearSVC(max_iter=10000)
    svm.fit(train_data, train_labels)
    calculate_errors("SVM", svm)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(train_data, train_labels)
    calculate_errors("Decision Tree", decision_tree)

    random_forest = RandomForestClassifier()
    random_forest.fit(train_data, train_labels)
    calculate_errors("Random Forest", random_forest)

    adaBoost = AdaBoostClassifier()
    adaBoost.fit(train_data, train_labels)
    calculate_errors("AdaBoost", adaBoost)

    boosting = GradientBoostingClassifier()
    boosting.fit(train_data, train_labels)
    calculate_errors("Gradient Boosting", boosting)

    lda = LinearDiscriminantAnalysis()
    lda.fit(train_data, train_labels)
    calculate_errors("Linear Discriminant Analysis", lda)

    mlp = MLPClassifier(max_iter=1000)
    mlp.fit(train_data, train_labels)
    calculate_errors("Multi-Layer Perceptron Regression", mlp)

    logistic = LogisticRegression()
    logistic.fit(train_data, train_labels)
    calculate_errors("Logistic Regression", logistic)