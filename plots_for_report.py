import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from my_functions import make_confusion_matrix

plt.rcdefaults()
fig, ax = plt.subplots()

# Lengthscales, RBF kernels in regression model
features = ('Weekly period', 'Free tests available', 'Assessments in week', 'Days symptomatic', 'Unique symptoms in week', 'Age', 'Contact health worker', 'Gender', 'BMI', 'Preconditions')
lengthscale = [1.00, 0.18, 0.24, 0.20, 0.50, 0.04, 0.07, 0.04, 0.05, 0.07]
lengthscale, features = zip(*sorted(zip(lengthscale, features), reverse=True))
y_pos = np.arange(len(features))
# performance = 3 + 10 * np.random.rand(len(people))
# error = np.random.rand(len(people))
ax.barh(y_pos, lengthscale, align='center')
ax.set_yticks(y_pos, labels=features)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Lengthscale')
ax.set_title('Normalised lengthscales of feature kernels in GP regression model', loc='right')
plt.tight_layout()
plt.show()


# Lengthscales, RBF kernels in GP classification model
plt.rcdefaults()
fig, ax = plt.subplots()
features = ('Weekly period', 'Free tests available', 'Assessments in week', 'Days symptomatic', 'Unique symptoms in week', 'Age', 'Contact health worker', 'Gender', 'BMI', 'Preconditions')
lengthscale = [22.51, 2.16, 4.01, 16.00, 0.17, 0.73, 0.66, 0.81, 0.76, 1.10]
lengthscale_norm = [x / max(lengthscale) for x in lengthscale]
lengthscale_norm, features = zip(*sorted(zip(lengthscale_norm, features), reverse=True))
y_pos = np.arange(len(features))
# performance = 3 + 10 * np.random.rand(len(people))
# error = np.random.rand(len(people))
ax.barh(y_pos, lengthscale_norm, align='center')
ax.set_yticks(y_pos, labels=features)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Lengthscale')
ax.set_title('Normalised lengthscales of feature kernels in GP classification model', loc='right')
plt.tight_layout()
plt.show()



# Confusion matrix for RF classifier
cf_matrix = np.array([[2226, 634],[694, 273]])
print(cf_matrix)

make_confusion_matrix(cf_matrix,
                       group_names=['True Neg', 'False Pos','False Neg','True Pos'],
                       categories='auto',
                       count=True,
                       percent=True,
                       cbar=False,
                       xyticks=True,
                       xyplotlabels=True,
                       sum_stats=True,
                       figsize=None,
                       cmap='Blues',
                       title='Confusion matrix for Random Forest classifier')




# Confusion matrix for GP classifier
cf_matrix = np.array([[5934, 610],[1410, 2021]])
print(cf_matrix)

make_confusion_matrix(cf_matrix,
                       group_names=['True Neg', 'False Pos','False Neg','True Pos'],
                       categories='auto',
                       count=True,
                       percent=True,
                       cbar=False,
                       xyticks=True,
                       xyplotlabels=True,
                       sum_stats=True,
                       figsize=None,
                       cmap='Blues',
                       title='Confusion matrix for Gaussian Process classifier')


#
#
# # Example data
# features = ('Age category', 'Preconditions', 'Contact health worker', 'Sex', 'Underweight', 'Healthy', 'Overweight', 'Obese')
# mean_importance = [0.41, 0.03, 0.05, 0.08, 0.03, 0.31, 0.05, 0.04]
# error = [0.05, 0.02, 0.025, 0.045, 0.02, 0.09, 0.045, 0.035]
#
# y_pos = np.arange(len(features))
# # performance = 3 + 10 * np.random.rand(len(people))
# # error = np.random.rand(len(people))
#
# ax.barh(y_pos, mean_importance, xerr=error, align='center')
# ax.set_yticks(y_pos, labels=features)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Relative importance')
# ax.set_title('Predictive importance of features in classification model')
# plt.tight_layout()
# plt.show()

