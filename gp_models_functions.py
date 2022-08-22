import GPy
import numpy as np
import math
import gpflow

from GPy import kern as kernels
from collections import Counter, defaultdict
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# from Evaluation import eval_classification


class ModelsClass_GPy(object):
    def __init__(self, x, y, x_=None, y_=None, kernel=None, likelihood=None, mean_func=None):
        self.x = x
        self.y = y
        self.x_ = x_
        self.y_ = y_
        self.kernel = kernel
        self.likelihood = likelihood
        self.mean_func = mean_func

    def simple_regression(self):
        """
        Simple regression model which considers samples independently of each other
        :return: optimised model and dictionary with model predictions
        """
        simple_model = {"Testing": [], "Model": [], "Real_data": []}#, "Likelihood": [], "Density_prob": []}

        # Check the dimensionality of the data
        dim = np.shape(self.x)

        # Set the mean function
        if self.mean_func is None:
            mean_func = None
        else:
            mean_func = self.mean_func

        # Set up the kernels, relative to where the features are in the data being passed to the model
        # Linear kernels for features with binary values
        # Squared exponential kernels for continuous/discrete

        if self.kernel is None:

            k_period = kernels.RBF(input_dim=1, active_dims=[0])
            k_free = kernels.Linear(input_dim=1, active_dims=[1])
            k_assessments = kernels.RBF(input_dim=1, active_dims=[2])
            k_days_symptomatic = kernels.RBF(input_dim=1, active_dims=[3])
            k_symptoms_in_period = kernels.RBF(input_dim=1, active_dims=[4])
            k_age = kernels.RBF(input_dim=1, active_dims=[5])
            k_health_worker = kernels.Linear(input_dim=1, active_dims=[6])
            k_gender = kernels.Linear(input_dim=1, active_dims=[7])
            k_bmi = kernels.RBF(input_dim=1, active_dims=[8])
            k_preconditions = kernels.Linear(input_dim=1, active_dims=[9])
            kernel = k_period + k_free + k_assessments + k_days_symptomatic + k_symptoms_in_period + k_age + k_health_worker + k_gender + k_bmi + k_preconditions

        else:
            kernel = self.kernel

        # Define the simple model
        m = GPy.models.GPRegression(X=self.x,
                                    Y=self.y,
                                    kernel=kernel,
                                    normalizer=False)

        # Model optimisation
        m.optimize('bfgs', max_iters=250)

        print(m)

        prediction, variance = m.predict(self.x_)
        simple_model["Testing"].append(prediction)
        simple_model["Real_data"].append(self.y_)
        simple_model["Model"].append(m)

        # Add evaluation if needed?

        return simple_model, m

def simple_classification(x, y):
    """
    Simple classifier: GPC with Bernoulli likelihood and a Sparse optimisation
    :return: optimised model and dict with the model predictions
    """
    simple_model = {"Testing": np.array([]),
                    "Model": [],
                    "Real_data": np.array([]),
                    "Likelihood": np.array([]),
                    "Density_prob": np.array([]),
                    'ID_test': np.array([])}

    # self.likelihood = gpflow.likelihoods.Bernoulli()

    # Kernel
    # k_period = kernels.RBF(input_dim=1, active_dims=[0])
    # k_free = kernels.Linear(input_dim=1, active_dims=[1])
    # k_assessments = kernels.RBF(input_dim=1, active_dims=[2])
    # k_days_symptomatic = kernels.RBF(input_dim=1, active_dims=[3])
    # k_symptoms_in_period = kernels.RBF(input_dim=1, active_dims=[4])
    # k_age = kernels.RBF(input_dim=1, active_dims=[5])
    # k_health_worker = kernels.Linear(input_dim=1, active_dims=[6])
    # k_gender = kernels.Linear(input_dim=1, active_dims=[7])
    # k_bmi = kernels.RBF(input_dim=1, active_dims=[8])
    # k_preconditions = kernels.Linear(input_dim=1, active_dims=[9])
    # kernel = k_period + k_free + k_assessments + k_days_symptomatic + k_symptoms_in_period + k_age + k_health_worker + k_gender + k_bmi + k_preconditions

    kernel = kernels.RBF(input_dim=10, active_dims=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ARD=True)

    # Model definition
    m = GPy.models.GPClassification(x, y, kernel=kernel)

    # Model optimisation
    m.optimize('bfgs', max_iters=200)

    print(m)

    # prediction = m.predict(x_)[0]
    # print(prediction)
    # simple_model["Testing"].append(prediction)
    # simple_model["Real_data"].append(y_)
    # simple_model["Model"].append(m)

    # Add evaluation if needed?

    return simple_model, m


def regression_evaluation(data):
    """
    Takes a dictionary of expected and predicted outputs from a model, 
    and displays/returns evaluation metrics
    """
    if type(data) is not dict:
        raise TypeError('Both the predictions and real data should have been save in a dictionary structure.')

    if 'Testing' not in data.keys():
        raise ValueError('Missing the testing samples.')

    if 'Real_data' not in data.keys():
        raise ValueError('Missing the real data labels.')

    if data['Real_data'].__len__() != data['Testing'].__len__():
        raise ValueError('The observed data and the predictions need to have the same size!')

    evaluation = {'r2': [], 'MAE': [], 'MSE': [], 'RMSE':[]}

    # Get predicted and actual values
    predicted = data['Testing'][0]
    predicted = [float(i) for i in predicted]
    actual = [float(i) for i in data['Real_data'][0]]

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mse)

    evaluation['r2'].append(r2)
    evaluation['MAE'].append(mae)
    evaluation['MSE'].append(mse)
    evaluation['RMSE'].append(rmse)

    print('r2:', r2)
    print('mean absolute error:', mae)
    print('mean squared error', mse)
    print('RMSE:', rmse)

    return evaluation

def return_test_total_class(x):
    if x == 0:
        return 0
    elif 1 <= x < 3:
        return 1
    else:
        return 2

def return_test_class(x):
    if x == 0:
        return 0
    else:
        return 1

# class ModelsClass(object):
#     def __init__(self, x, y, data_test=None, kernel=None, likelihood=None, mean_func=None):
#         """
#         Classification models:
#             -> Naive Model
#             -> Hierarchical model
#             -> Coregionalised and Heteroscedastic model
#         :param x: Features matrix - training set. N x D, where N is the number of samples and D the number of features.
#         :param y: Labels vector - training set. N x C, where N is the number of samples. C can be either 1,
#         when only simple models are considered or C>1 for heteroscedastic and coregionalised models.
#         :param kernel: Kernel structure to be used in the model.
#         :param likelihood: Likelihood function to be considered by the model.
#         :param mean_func: Mean prior function to be encoded by the model.
#         """
#         self.x = x
#         self.y = y
#
#         if data_test is None:
#
#             self.x_ = None
#             self.y_ = None
#
#             # raise Warning("The model will not be validated using an independent validation set. Only model training "
#             #               "will be executed.")
#
#         else:
#             x_, y_ = data_test
#             self.x_ = x_
#             self.y_ = y_
#
#             if x.shape[1] is not x_.shape[1]:
#                 raise ValueError("The matrix of testing features needs to have the same shape of the training data.")
#             if y.shape[1] is not y_.shape[1]:
#                 raise ValueError("The vector of testing labels requires to have the same shape of the testing set.")
#
#         if kernel is None:
#             self.dim_symp = np.arange(7, 20)
#                 # # Creates a vector of the dimension of the symptoms to
#             #[x + 7 for x in [9, 0, 11, 3, 6, 10, 7, 8, 1, 4]]
#             # Initialise the vector with the length scales to be used in the kernel encoding the subjects symptoms
#             self.length_symp = tf.convert_to_tensor(np.ones_like(self.x[0, self.dim_symp]), dtype=default_float())
#
#             # Initialise the vector with the length scales to be used in the kernel encoding the subjects pre-conditions
#             self.length_PC = tf.convert_to_tensor(np.ones_like(self.x[0, np.arange(3, 7)]), dtype=default_float())
#
#             # Defines the time points vector. It adds one to account for the encoding of the baseline with 0
#             #self.x[:, [21]] += 1
#
#             self.k_age = kernels.SquaredExponential(active_dims=[0])
#             self.k_bmi = kernels.SquaredExponential(active_dims=[1])
#             self.k_PC = kernels.ArcCosine(order=1, weight_variances=self.length_PC,
#                                           active_dims=np.arange(3, 7))
#
#             k_gender = kernels.SquaredExponential(active_dims=[2])
#             k_HCW = kernels.SquaredExponential(active_dims=[22])
#
#             k_symp = kernels.ArcCosine(order=2, weight_variances=self.length_symp,
#                                                 active_dims=self.dim_symp)
#
#             # Kernel encoding the demographic information
#             k_demo = self.k_age + k_gender
#
#             # Kernel encoding the effect of the health pre-conditions and symptoms
#             self.k_cond = self.k_PC * self.k_bmi
#
#             # Final kernel considering both demographic information and the pre-conditions
#             self.kernel = k_demo + (self.k_cond * k_symp) + k_HCW
#
#         else:
#             self.kernel = kernel
#
#         if likelihood is None:
#             # It defines a logit function instead of probit
#             def inv_link(x):
#                 jitter = 1e-2  # ensures output is strictly between 0 and 1
#                 return (tf.exp(x)/(tf.exp(x)+1)) * (1 - 2 * jitter) + jitter
#
#             self.likelihood = gpflow.likelihoods.Bernoulli(invlink=inv_link)
#         else:
#             self.likelihood = likelihood
#
#         if mean_func is None:
#             self.mean_func = 0
#         else:
#             self.mean_func = mean_func


#
# class CrossEval:
#     """
#     Cross-validation Schemes.
#     All the cross-validation schemes require the group information for stratification purposes.
#     """
#     def __init__(self, x, y, groups=None):
#         self.X = x
#         self.Y = y
#
#         if groups is None:
#             raise Warning('It only can perform cross-validation with group dependency.')
#
#         else:
#             self.groups = groups
#
#     def StratifiedYFold(self, n_split=10):
#
#         lpgo = RepeatedStratifiedGroupKFold(n_splits=n_split)
#
#         # Initialise training set
#         x = []
#         y = []
#         # Initialise testing set
#         x_ = []
#         y_ = []
#
#         Y = self.Y.astype(int)
#         g = self.groups.astype(int)
#         for train_index, test_index in lpgo.split(X=self.X, y=np.reshape(Y, [self.Y.shape[0], ]),
#                                                   groups=np.reshape(g, [self.groups.shape[0], ])):
#             print(np.intersect1d(train_index, test_index))
#             x.append(self.X[train_index, :])
#             y.append(self.Y[train_index, :])
#             x_.append(self.X[test_index, :])
#             y_.append(self.Y[test_index, :])
#
#         return x, y, x_, y_
#
# class RepeatedStratifiedGroupKFold:
#
#     def __init__(self, n_splits=5, n_repeats=1, random_state=1338):
#         self.n_splits = n_splits
#         self.n_repeats = n_repeats
#         self.random_state = random_state
#
#     # Implementation based on this kaggle kernel:
#     #    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
#     def split(self, X, y=None, groups=None):
#         k = self.n_splits
#
#         def eval_y_counts_per_fold(y_counts, fold):
#             y_counts_per_fold[fold] += y_counts
#             std_per_label = []
#             for label in range(labels_num):
#                 label_std = np.std(
#                     [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
#                 )
#                 std_per_label.append(label_std)
#             y_counts_per_fold[fold] -= y_counts
#             return np.mean(std_per_label)
#
#         rnd = check_random_state(self.random_state)
#         for repeat in range(self.n_repeats):
#             labels_num = np.max(y) + 1
#             y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
#             y_distr = Counter()
#             for label, g in zip(y, groups):
#                 y_counts_per_group[g][label] += 1
#                 y_distr[label] += 1
#
#             y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
#             groups_per_fold = defaultdict(set)
#
#             groups_and_y_counts = list(y_counts_per_group.items())
#             rnd.shuffle(groups_and_y_counts)
#
#             for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
#                 best_fold = None
#                 min_eval = None
#                 for i in range(k):
#                     fold_eval = eval_y_counts_per_fold(y_counts, i)
#                     if min_eval is None or fold_eval < min_eval:
#                         min_eval = fold_eval
#                         best_fold = i
#                 y_counts_per_fold[best_fold] += y_counts
#                 groups_per_fold[best_fold].add(g)
#
#             all_groups = set(groups)
#             for i in range(k):
#                 train_groups = all_groups - groups_per_fold[i]
#                 test_groups = groups_per_fold[i]
#
#                 train_indices = [i for i, g in enumerate(groups) if g in train_groups]
#                 test_indices = [i for i, g in enumerate(groups) if g in test_groups]
#
#                 yield train_indices, test_indices