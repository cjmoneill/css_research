import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from GPy import kern as kernels
from gp_models_functions import ModelsClass_GPy, regression_evaluation, return_test_total_class, simple_classification, return_test_class
from my_functions import stratify_ages
from GPy import plotting as plot
from sklearn import metrics

def main(path):

# Read in and tidy up data to be passed to the model
# Read in and create dataframe
    dataframe = pd.read_csv(path)
    print('unique patients in df:', dataframe['id_patients'].nunique())
    print('Initial dataframe keys:', dataframe.keys())

    # Tidy up NaNs in Healthcare Worker status
    dataframe['health_worker_status'] = dataframe['health_worker_status'].fillna(0)

    # Add a column with classes for weekly test totals
    dataframe['weekly_test_class'] = dataframe['tests_in_period'].apply(lambda x: return_test_total_class(x))

    # Add column with binary classification for classification model
    dataframe['weekly_test_class_binary'] = dataframe['tests_in_period'].apply(lambda x: return_test_class(x))

    # Add a column with age categories
    dataframe = stratify_ages(dataframe)
    # Drop incomplete week data (i.e. week [7])
    dataframe = dataframe[dataframe.period_relative_to_change < 7]

    print(dataframe.keys())
    print(dataframe.period_relative_to_change.head(50))

    # Try version with 4 tests and above only
    # dataframe_4_tests = dataframe[dataframe['tests_in_period'] >= 4]
    # dataframe = dataframe_4_tests

    # Print basic info
    print('Unique patients in data:', dataframe['id_patients'].nunique())
    print('Unique columns/rows in data:', dataframe.shape)

    # Select columns for model features
    model_features = ['period_relative_to_change', 'tests_free_during_period', 'assessments_in_period',
                      'days_symptomatic', 'unique_symptoms_total', 'age', 'health_worker_status',
                      'gender', 'bmi_range', 'preconditions_status']
    model_targets = ['weekly_test_class_binary']
    print('Model features:', model_features)
    print('Model targets:', model_targets)

    # Create arrays to pass into model
    X = dataframe.filter(items=model_features).to_numpy()
    Y = dataframe.filter(items=model_targets).to_numpy()
    print('X shape:', X.shape)
    print('y shape', Y.shape)

    # Define the cross-validation scheme
    # Just do randomly to start with... should be stratified (although at least patients will not be in both groups here)
    X_split = int(len(X) / 2)
    Y_split = int(len(Y) / 2)
    x = X[:X_split,:]
    x_ = X[X_split:,:]
    y = Y[:Y_split,:]
    y_ = Y[Y_split:,:]

    print('train min max:', min(y), max(y))
    print('test min max:', min(y_), max(y_))

    # x, y, x_, y_ = CrossEval(X, y, groups=group).StratifiedYFold(n_split=5)

    print('shape:', x.shape, y.shape, x_.shape, y_.shape)

    print('Optimising the Model')
    naive_model = {"Testing": [], "Model": [], "Real_data": [], "Likelihood": [], "Density_prob":[], "ID_test":[]}

    # Loop passes sample individually
    # for j in range(len(x)):
    #     print('------Iteration------: ', j)
    #     print('Model naive 1')
    #     x_sample = np.expand_dims(x[j], axis=0)
    #     y_sample = np.expand_dims(y[j], axis=0).reshape(-1,1)
    #     x__sample = np.expand_dims(x_[j], axis=0)
    #     y__sample = np.expand_dims(y_[j], axis=0).reshape(-1,1)
    #     naive_model = ModelsClass_GPy(x_sample, y_sample, x__sample, y__sample).simple_regression()
    #
    #     for k, v in naive_model.items():
    #         naive_model[k].append(v[0])

    # Test with sample of rows
    x = X[:9975,:]
    x_ = X[:9975,:]
    y = Y[:9975,:]
    y_ = Y[:9975,:]

    print('shape:', x.shape, y.shape, x_.shape, y_.shape)

    ### Regression models
    # Simple kernel (all features)
    # naive_model_, m = ModelsClass_GPy(x, y, x_, y_, kernel=kernels.RBF(input_dim=10, active_dims=[0,1,2,3,4,5,6,7,8,9], ARD=True)).simple_regression()

    # RBF for all features
    # k_period = kernels.RBF(input_dim=1, active_dims=[0])
    # k_free = kernels.RBF(input_dim=1, active_dims=[1])
    # k_assessments = kernels.RBF(input_dim=1, active_dims=[2])
    # k_days_symptomatic = kernels.RBF(input_dim=1, active_dims=[3])
    # k_symptoms_in_period = kernels.RBF(input_dim=1, active_dims=[4])
    # k_age = kernels.RBF(input_dim=1, active_dims=[5])
    # k_health_worker = kernels.RBF(input_dim=1, active_dims=[6])
    # k_gender = kernels.RBF(input_dim=1, active_dims=[7])
    # k_bmi = kernels.RBF(input_dim=1, active_dims=[8])
    # k_preconditions = kernels.RBF(input_dim=1, active_dims=[9])
    # kernel = k_period + k_free + k_assessments + k_days_symptomatic + k_symptoms_in_period + k_age + k_health_worker + k_gender + k_bmi + k_preconditions
    # naive_model_, m = ModelsClass_GPy(x, y, x_, y_, kernel=kernel).simple_regression()

    # Designed kernel (uses one specified in method)
    # naive_model_, m = ModelsClass_GPy(x, y, x_, y_).simple_regression()

    # Grouped kernels (kernels grouped by 'theme')
    # k_before_after = kernels.RBF(input_dim=2, active_dims=[0,1])
    # k_assessments = kernels.RBF(input_dim=1, active_dims=[2])
    # k_symptoms = kernels.RBF(input_dim=2, active_dims=[3, 4])
    # k_demographics = kernels.RBF(input_dim=4, active_dims=[5,7,8,9])
    # k_health_worker = kernels.RBF(input_dim=1, active_dims=[6])
    # kernel = k_before_after + k_assessments + k_symptoms + k_demographics + k_health_worker
    # naive_model_, m = ModelsClass_GPy(x, y, x_, y_, kernel=kernel).simple_regression()

    ### Classification models
    naive_model_, m = simple_classification(x, y)

    # Classification evaluation

    predictions = m.predict((x_))
    predictions = predictions[0]
    predictions = np.where(predictions > 0.5, 1, 0)
    actual_labels = y_
    print(predictions)
    print(actual_labels)

    print('accuracy:', metrics.accuracy_score(actual_labels, predictions))
    print('precision:', metrics.precision_score(actual_labels, predictions))
    print('roc auc:', metrics.roc_curve(actual_labels, predictions))

    tn, tp, fn,fp = metrics.confusion_matrix(actual_labels, predictions).ravel()
    specificity = tn / (tn+fp)
    print('specificity:', specificity)
    sensitivity = tp / (tp+fn)
    print('sensitivity:', sensitivity)
    conf_matrix = metrics.confusion_matrix(actual_labels, predictions).ravel()
    print('confusion matrix:', conf_matrix)

# Regression evaluation

    # for k, v in naive_model_.items():
    #     naive_model[k].append(v[0])
    #
    # naive_model_evaluation = regression_evaluation(naive_model)
    # print(naive_model_evaluation)
    #
    # # Plot the predicted against the actual
    # predicted = [float(i) for i in naive_model['Testing'][0]]
    # actual = [float(i) for i in naive_model['Real_data'][0]]
    #
    # plt.plot(actual, predicted, 'ro')
    # plt.xlabel('Actual')
    # plt.ylabel('Predicted')
    # plt.show()

    # Plot the ARD representation
    lengthscale_array = m['rbf.lengthscale'][:]
    print('rbf le array:', lengthscale_array)


    # # Predict for some patients with known number of tests (from 0 tests to 7)
    # # Patient with 0 tests
    # print('Sample with 0 tests:')
    # sample = X[5140,:].reshape(1,10)
    # print(sample.shape)
    # print('Features:', sample)
    # print('Label:', Y[5140, :])
    # prediction, variance = m.predict(sample)
    # print('Predicted:', prediction)
    #
    # # Patient with 1 test
    # print('Sample with 1 tests:')
    # sample = X[5174,:].reshape(1,10)
    # print('Features:', sample)
    # print('Label:', Y[5174, :])
    # prediction, variance = m.predict(sample)
    # print('Predicted:', prediction)
    #
    # # Patient with 6 tests
    # print('Sample with 6 tests:')
    # sample = X[4929, :].reshape(1, 10)
    # print('Features:', sample)
    # print('Label:', Y[4929, :])
    # prediction, variance = m.predict(sample)
    # print('Predicted:', prediction)
    #
    # # Patient with 7 tests
    # print('Sample with 7 tests:')
    # sample = X[246, :].reshape(1, 10)
    # print('Features:', sample)
    # print('Label:', Y[246, :])
    # prediction, variance = m.predict(sample)
    # print('Predicted:', prediction)

    print('done')
    print('done')
if __name__ == '__main__':
    main(path = '/nvme1_mounts/nvme1lv02/coneill/project_v4/temporal_complete_df')
