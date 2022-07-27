import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from my_functions import convert_df_dates
from my_functions import compare_tests_around_policy_change
from my_functions import EvaluatePerformance
from my_functions import EvaluatePerformanceCV
from my_functions import features_dataframe
from my_functions import plot_search_results

def main(train, test, start_date, end_date, policy_change_date):

    # Read into dataframes
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)
    print(train_df.keys())
    print(test_df.keys())

    # Convert 'nan' to 0 for contact_health_worker
    train_df['health_worker_status'] = train_df.health_worker_status.replace(np.nan, 0)
    test_df['health_worker_status'] = test_df.health_worker_status.replace(np.nan, 0)

    # Demographics
    print('Train sample demographics:')
    print(train_df.country.value_counts())
    print(train_df.age_category.value_counts())
    print(train_df.health_worker_status.value_counts())
    print(train_df.gender.value_counts())
    print(train_df.bmi_range.value_counts())
    print(train_df.preconditions_status.value_counts())
    print(train_df.lower_higher.value_counts())

    print('Test sample demographics:')
    print(test_df.country.value_counts())
    print(test_df.age_category.value_counts())
    print(test_df.health_worker_status.value_counts())
    print(test_df.gender.value_counts())
    print(test_df.bmi_range.value_counts())
    print(test_df.preconditions_status.value_counts())
    print(test_df.lower_higher.value_counts())

    print(train_df.head(20))
    print(test_df.head(20))

    # Split into feature and target vectors
    X_train = train_df.filter(items=['country', 'age_category', 'health_worker_status',
                                                  'gender', 'bmi_range', 'preconditions_status',
                                                  ]).to_numpy()
    y_train = train_df.filter(items=['lower_higher']).to_numpy()
    y_train = np.squeeze(y_train)

    X_test =  test_df.filter(items=['country', 'age_category', 'health_worker_status',
                                                  'gender', 'bmi_range', 'preconditions_status',
                                                  ]).to_numpy()
    y_test = test_df.filter(items=['lower_higher']).to_numpy()
    y_test = np.squeeze(y_test)

    # Print shapes
    print('X train shape:', X_train.shape)
    print('y train shape:', y_train.shape)
    print('X test shape:', X_test.shape)
    print('y test shape:', y_test.shape)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)


    ### Random forest classifier

    print('Random forest classifier')
    rnd_clf = RandomForestClassifier(random_state=42)

    # Training performance

    # Number of trees in the random forest
    estimators = [10,100]#500]
    # Criterion
    criterion = ["gini"]
    # Number of features at every split
    max_features = ['sqrt', 'log2']
    # Maximum number of levels in a tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 5, 10]
    # Method of selecting samples
    bootstrap = [True, False]

    param_grid = dict(n_estimators=estimators,
                      criterion=criterion,
                      max_features=max_features,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf,
                      bootstrap=bootstrap
    )

    rnd_grid = GridSearchCV(rnd_clf, param_grid, cv=5)
    rnd_grid = rnd_grid.fit(X_train, y_train)

    print('Best CV accuracy:', rnd_grid.best_score_)
    print('Best parameters:')
    print(rnd_grid.best_params_)

    tuned_rnd_clf = rnd_grid.best_estimator_

    # Evaluate performance on training data
    EvaluatePerformance(tuned_rnd_clf, X_train, y_train, modeltitle="Random forest classifier")

    # Evaluate performance with cross-validation
    EvaluatePerformanceCV(tuned_rnd_clf, X_train, y_train, modeltitle="Random forest classifier")

    # Selecting features from model
    selected_features = tuned_rnd_clf.feature_importances_
    print(selected_features)

    # Evaluate on test data
    print('Performance on test data:')
    EvaluatePerformance(tuned_rnd_clf, X_test, y_test, modeltitle="Random forest classifier")

    # ROC AUC
    y_score = tuned_rnd_clf.predict_proba(X_test)[:,1]
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_score)
    print('roc_auc_score for classifier: ', roc_auc_score(y_test, y_score))

    plt.subplots(1, figsize=(10, 10))
    plt.title('Receiver Operating Characteristic - DecisionTree')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Calculate Youden's J Statistic
    # J = sensitivity + specificity - 1
    y_pred = tuned_rnd_clf.predict(y_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    true_neg = conf_matrix[0][0]
    false_neg = conf_matrix[1][0]
    true_pos = conf_matrix[1][1]
    false_pos = conf_matrix[0][1]

    youden_j = (true_pos / (true_pos + false_neg)) + (true_neg / (true_neg + false_pos)) - 1
    print('Youden J statistic:', youden_j)


    # Visualise comparative performance
    plot_search_results(rnd_grid)


if __name__ == '__main__':
    main(train = '/nvme1_mounts/nvme1lv02/coneill/project_v4/stratified_train_sample.csv',
         test = '/nvme1_mounts/nvme1lv02/coneill/project_v4/stratified_test_sample.csv',
         start_date = datetime.date(2022, 2, 1),
         end_date = datetime.date(2022,5,30),
         policy_change_date = datetime.date(2022,4,1))


    # ### Training and tuning the classifier
    #
    # print('Basic logistic regression model:')
    # print('\n')
    # model = LogisticRegression()
    # model = model.fit(X_train, y_train)
    #
    # print('Performance on training dataset:')
    # EvaluatePerformance(model, X_train, y_train, modeltitle="Logistic Regression")
    # print('Performance on test dataset:')
    # EvaluatePerformance(model, X_test, y_test, modeltitle="Logistic Regression")
    #
    # print('Tuned logistic regression model')
    # model2 = LogisticRegression()
    # c_values = np.logspace(-3, 3, 13)
    # penalty = ['l2']
    # solvers = ['lbfgs', 'liblinear']
    # param_grid = dict(solver=solvers, penalty=penalty, C=c_values)
    # gridsearch = GridSearchCV(model2, param_grid, cv=5)
    # gridsearch = gridsearch.fit(X_train, y_train)
    # tuned_lr_model = gridsearch.best_estimator_
    #
    # print('Performance on training dataset:')
    # EvaluatePerformance(tuned_lr_model, X_train, y_train, modeltitle="Tuned logistic Regression")
    # print('Best C: ', tuned_lr_model.C)
    # print('Best params: ', gridsearch.best_params_ )
    # print("\n")
    #
    # means = gridsearch.cv_results_['mean_test_score']
    # stds = gridsearch.cv_results_['std_test_score']
    # params = gridsearch.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    #
    # print('Performance on test dataset:')
    # EvaluatePerformance(tuned_lr_model, X_test, y_test, modeltitle="Tuned logistic Regression")
    # print("\n")
    #
    # # Odds ratio
    # print('Odds ratios of best model:')
    # odds_ratios = np.exp(tuned_lr_model.coef_)
    # print(odds_ratios)