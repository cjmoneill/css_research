import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.setrecursionlimit(3000)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from my_functions import convert_df_dates
from my_functions import compare_tests_around_policy_change
from my_functions import features_dataframe
from my_functions import EvaluatePerformance
from my_functions import stratified_sample_requirements
from my_functions import stratify_ages
from my_functions import return_stratified_test_sample

def main(path, start_date, end_date, policy_change_date):

    # Read in and create dataframe
    dataframe = pd.read_csv(path)
    print(dataframe.head(20))
    print(dataframe.keys())
    dataframe = convert_df_dates(dataframe)

    # Create dataframe for target values
    dataframe_targets = compare_tests_around_policy_change(dataframe, start_date, end_date, policy_change_date)
    print(dataframe_targets)

    # Combine higher & stable into the same class
    dataframe_targets['lower_higher'] = dataframe_targets.lower_stable_higher.replace({2: 1})
    print('lower higher targets:')

    # Select the columns for the target variable
    dataframe_targets = dataframe_targets.filter(items=['id_patients', 'lower_higher'])
    targets_head = dataframe_targets.head(20)
    print('targets:')
    print(dataframe_targets)

    # Create dataframe for features
    dataframe_features = features_dataframe(dataframe)

    # Merge to ensure the order is the same
    dataframe_merged = dataframe_features.merge(dataframe_targets, how='inner', on='id_patients')
    print('Merged dataframe:')
    print(dataframe_merged)
    print('Unique patients:', dataframe_merged['id_patients'].nunique())
    print('Dataframe length', len(dataframe_merged))

   # Split into test and train datasets, stratified according to the features and classification

    # Add a column with age categories
    dataframe_merged = stratify_ages(dataframe_merged)

    # First get the ideal totals for each class in a test population
    stratified_sample_requirements(dataframe_merged, train_percentage=0.8, test_percentage=0.2)

    # Create dictionaries containing the ideal number of patients for each class to be sampled in the test set
    required_ages = {0: 10, 1: 197, 2: 743, 3: 1970, 4: 907}
    required_genders = {0: 2378, 1:1451}
    required_hw = {0: 1557, 1: 33}
    required_precs = {0: 3099, 1: 730}
    required_bmi = {0: 51, 1: 1729, 2: 1323, 3: 725}
    required_targets = {0: 2885, 1: 944}

    test_sample = return_stratified_test_sample(dataframe_merged, required_ages, required_genders, required_hw,
                                                required_precs, required_bmi, required_targets)

    test_dataframe = return_stratified_test_sample(dataframe_merged, required_ages, required_genders,required_hw,
                                                    required_precs, required_bmi, required_targets)


    # Drop country data (not relevant if only using England)
    dataframe_merged = dataframe_merged.drop(columns=['country'])

    df_head = dataframe_merged.head(20)
    print(df_head)
    # Drop patients who are not male/female
    # genders_to_exclude = [3, 4]
    # dataframe_merged = dataframe_merged[dataframe_merged.gender.isin(genders_to_exclude) == False]
    # counts_df, chemo_df, contact_df, asthmatics_df, bmi_df, gender_df = demographic_info(dataframe_merged)
    # print(gender_df)

    # One hot encode categorical columns where not binary
    # Replace numeric values with labels
    dataframe_merged['bmi_category'] = dataframe_merged.bmi_range.replace({0: 'underweight', 1: 'healthy_weight', 2: 'overweight', 3: 'obese'})
    # One hot encode
    bmi_one_hot = pd.get_dummies(dataframe_merged['bmi_category'])
    dataframe_merged = pd.concat([dataframe_merged, bmi_one_hot], axis=1)
    dataframe_merged = dataframe_merged.drop(columns = ['bmi_range', 'bmi_category'])

    # Print first 200 rows to evaluate
    sample_dataframe = dataframe_merged.head(n=200)
    print('First 200 rows:')
    print(sample_dataframe)

    # Print assessment of target labels (i.e. total testing more/same/less)
    target_totals = dataframe_merged['lower_higher'].value_counts()
    print('Totals (0: fewer after policy change, 1: same or higher')
    print(target_totals)

    # Create the feature matrix and label vector
    X = dataframe_merged.drop(columns=['id_patients', 'lower_higher'])
    X = X.to_numpy()
    print(X)
    print('X shape:', X.shape)

    y = dataframe_merged.filter(['lower_higher'])
    y = y.to_numpy()
    y = np.squeeze(y)

    print('y shape:', y.shape)
    print(y)
    # y = y[:, 1]
    # print('y shape:', y.shape)

    # Preprocessing

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Visualise with PCA
    pca_model = PCA(n_components=2)
    pca_model.fit(X)
    X_reduced = pca_model.transform(X)

    plt.plot(X_reduced[:, 0][y == 0], X_reduced[:, 1][y == 0], "r*", alpha=0.5, label='Lower')
    plt.plot(X_reduced[:, 0][y == 1], X_reduced[:, 1][y == 1], "bo", alpha=0.5, label='Higher/stable')
    # plt.plot(X_reduced[:, 0][y == 2], X_reduced[:, 1][y == 2], "g", alpha=0.5, label='Higher')
    plt.legend()
    plt.xlabel('Principle component 1')
    plt.ylabel('Principle component 2')
    plt.title('Dimensionality reduction')
    plt.show()


    # Stratify split (on label first... features later??)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('X train shape:', np.shape(X_train))
    print('y train shape:', np.shape(y_train))
    print('X test shape:', np.shape(X_test))
    print('y test shape:', np.shape(y_test))

    ### Training and tuning the classifier

    print('Basic logistic regression model:')
    print('\n')
    model = LogisticRegression()
    model = model.fit(X_train, y_train)

    print('Performance on training dataset:')
    EvaluatePerformance(model, X_train, y_train, modeltitle="Logistic Regression", test_or_train="train")
    print('Performance on test dataset:')
    EvaluatePerformance(model, X_test, y_test, modeltitle="Logistic Regression", test_or_train="test")

    print('Tuned logistic regression model')
    model2 = LogisticRegression()
    c_values = np.logspace(-3, 3, 13)
    penalty = ['l2']
    solvers = ['lbfgs', 'liblinear']
    param_grid = dict(solver=solvers, penalty=penalty, C=c_values)
    gridsearch = GridSearchCV(model2, param_grid, cv=5)
    gridsearch = gridsearch.fit(X_train, y_train)
    tuned_lr_model = gridsearch.best_estimator_

    print('Performance on training dataset:')
    EvaluatePerformance(tuned_lr_model, X_train, y_train, modeltitle="Tuned logistic Regression", test_or_train="train")
    print('Best C: ', tuned_lr_model.C)
    print('Best params: ', gridsearch.best_params_ )
    print("\n")

    means = gridsearch.cv_results_['mean_test_score']
    stds = gridsearch.cv_results_['std_test_score']
    params = gridsearch.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print('Performance on test dataset:')
    EvaluatePerformance(tuned_lr_model, X_test, y_test, modeltitle="Tuned logistic Regression", test_or_train="test")
    print("\n")

    # Odds ratio
    print('Odds ratios of best model:')
    odds_ratios = np.exp(tuned_lr_model.coef_)
    print(odds_ratios)
    print(dataframe_features.keys())

    # Random forest classifier
    print('Random forest classifier')
    rnd_clf = RandomForestClassifier(class_weight='balanced', random_state=42)

    # Number of trees in the random forest
    estimators = [5,15,25]
    # Criterion
    criterion = "gini"
    # Number of features at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in a tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
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

    tuned_rnd_grid = rnd_grid.best_estimator_

    print(EvaluatePerformance(tuned_rnd_grid, X_train, y_train, modeltitle="Random forest classifier", test_or_train="Train"))
    print(EvaluatePerformance(tuned_rnd_grid, X_test, y_test, modeltitle="Random forest classifier", test_or_train="Test"))

    # Selecting features from model


    print('Support vector classifier model:')
    linear_svc_model = LinearSVC()
    linear_svc_model = linear_svc_model.fit(X_train, y_train)

    print('Performance on training dataset:')
    EvaluatePerformance(linear_svc_model, X_train, y_train, modeltitle="Linear SVC", test_or_train="train")
    print("\n")

    print('Performance on test dataset:')
    EvaluatePerformance(linear_svc_model, X_test, y_test, modeltitle="Linear SVC", test_or_train="test")
    print("\n")

    # SVC model with different kernels
    print('SVC models with different kernels:')

    kernel = ['linear', 'rbf', 'poly', 'sigmoid']
    for i in kernel:
        SVC_model = SVC(kernel=i, C=1)
        SVC_model = SVC_model.fit(X_train, y_train)
        print(EvaluatePerformance(SVC_model, X_train, y_train, '{} kernel'.format(i), test_or_train="train"))
        print(EvaluatePerformance(SVC_model, X_test, y_test, '{} kernel'.format(i), test_or_train="test"))
        print('\n')

    # print('SVC models with tuned parameters:')
    # linear_svc_model = SVC(kernel='linear')
    # parameter_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    # grid = GridSearchCV(linear_svc_model, parameter_grid, cv=5)
    # grid.fit(X_train, y_train)
    # tuned_SVC_linear_kernel = grid.best_estimator_
    #
    # EvaluatePerformance(tuned_SVC_linear_kernel, X_train, y_train, modeltitle="Tuned SVC with linear kernel")
    # print('Best C: ', round(tuned_SVC_linear_kernel.C, 2))
    # print("\n")
    #
    # EvaluatePerformance(tuned_SVC_linear_kernel, X_test, y_test, modeltitle="Tuned SVC with linear kernel")
    # print("\n")

    # Exploring sigmoid kernel
    sig_kernel_model = SVC(kernel='sigmoid')
    param_grid = {"C": np.logspace(-3, 3, 7)}
    grid_search = GridSearchCV(sig_kernel_model, cv=5, param_grid=param_grid)
    grid_search = grid_search.fit(X_train, y_train)

    print('Best C  :', grid_search.best_estimator_.C)
    print('Best CV accuracy:', round(grid_search.best_score_, 2))

    best_classifier = grid_search.best_estimator_

    print(EvaluatePerformance(best_classifier, X_train, y_train, modeltitle='best_estimator', test_or_train='train'))
    print(EvaluatePerformance(best_classifier, X_test, y_test, modeltitle='best_estimator', test_or_train='test'))


    # Random forest classifier
    print('Random forest classifier')
    rnd_clf = RandomForestClassifier(class_weight='balanced')

    estimators = [5,15,25]
    param_grid = dict(n_estimators=estimators)
    rnd_grid = GridSearchCV(rnd_clf, param_grid, cv=5)
    rnd_grid = rnd_grid.fit(X_train, y_train)
    tuned_rnd_grid = rnd_grid.best_estimator_

    print(EvaluatePerformance(tuned_rnd_grid, X_train, y_train, modeltitle="Random forest classifier", test_or_train="Train"))
    print(EvaluatePerformance(tuned_rnd_grid, X_test, y_test, modeltitle="Random forest classifier", test_or_train="Test"))

    # Selecting features from model



    # Odds ratio
    print('Odds ratios of best model:')
    odds_ratios = np.exp(best_classifier.coef_)
    print(odds_ratios)



if __name__ == '__main__':
    main('/nvme1_mounts/nvme1lv02/coneill/project_v4/merged_pat_tests_england.csv',

         start_date = datetime.date(2022, 2, 1),
         end_date = datetime.date(2022,5,30),
         policy_change_date = datetime.date(2022,4,1))