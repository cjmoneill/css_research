import pandas as pd
import datetime as dt
import numpy as np
import datetime
import statistics as stats
import matplotlib.pyplot as plt
import csv


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score

# Remove instances where BMI <10 or >95
def exclude_invalid_bmi(df):
    new_df = df
    new_df = new_df[(new_df.bmi >= 10)]
    new_df = new_df[(new_df.bmi < 95)]
    new_df = new_df.reset_index(drop=True)
    return new_df


# Convert dates function (for 'date_test' field)
def convert_df_dates(dataframe):
    # create new dataframe
    new_df = dataframe
    new_df['date_test'] = dataframe.date_test.replace('0', '2001, 1, 1')
    new_df['date_test'] = pd.to_datetime(new_df.date_test).dt.date
    return new_df

# Convert dates function (for specified series)
def convert_df_date_series(dataframe, series: str):
    new_df = dataframe
    new_df[series] = dataframe[series].replace('0', '2001, 1, 1')
    new_df[series] = pd.to_datetime(new_df[series]).dt.date
    return new_df

# Function for sliding window over a series to check against acceptance criteria
def assessment_frequency_criteria(series,
                                  start_date: datetime,
                                  end_date: datetime,
                                  period=7,
                                  ratio_of_weeks =2/3):

    # convert period to datetime delta, calculate periods
    period = datetime.timedelta(days=period)
    total_days = end_date - start_date
    full_periods = ((end_date - start_date) // period)
    last_period = total_days % period
    # set running date
    running_date = start_date
    weekly_totals = []
  # loop over the full periods, calculating assessment total in each window
    for i in range(full_periods):
        window_end = running_date + period
        total_in_window = sum(map(lambda x: x >= running_date and x < window_end, series))
        weekly_totals.append(total_in_window)
        running_date += period
  # add weighted tests for the remainder period
    if last_period.days != 0:
        total_in_last_window = (sum(map(lambda x: x >= (end_date - last_period) and x < end_date, series)))
        # print(end_date - last_period)
        # print('last period score:', total_in_last_window * period.days/last_period.days)
        weekly_totals.append(total_in_last_window * (period.days/last_period.days))
    # count the number of weeks where assessments is 2 or over
    count_over_2 = sum(map(lambda x : x >= 2, weekly_totals))
    # return True if 2 thirds or over periods have 2 assessmensts
    if count_over_2 / len(weekly_totals) >= ratio_of_weeks:
        return True
    else:
        return False

# Return a dataframe of patient ids and acceptance status
def return_dataframe_of_accepted_patients(dataframe,
                                          start_date=datetime.date(2022,1,1),
                                          end_date=datetime.date(2022,1,12),
                                          window=7,
                                          ratio_of_weeks= 2/3):
    # Group dataframe by patients
    grouped = dataframe.groupby('id_patients')
    # Apply the windowing function
    accepted = grouped['created_at_assessments'].apply(lambda x: assessment_frequency_criteria(x, start_date, end_date, window, ratio_of_weeks))
    print('accepted status:', accepted, type(accepted))
    # Create a dataframe with unique patient IDs
    # unique_patients = dataframe['id_patients'].unique()
    # print('uniques', unique_patients, type(unique_patients))
    # Add the accepted column

    # accepted_df = pd.DataFrame({'id_patients': pd.DataFrame(unique_patients), 'meets_criteria': pd.DataFrame(accepted)})
    # Liane's way:
    # accepted_df = pd.concat([pd.DataFrame(unique_patients), pd.DataFrame(accepted)], axis=0)
    # accepted_df.columns = ['id_patients', 'meets_criteria']

    # Try to merge instead
    accepted = accepted.to_frame()
    print(accepted.columns)
    accepted = accepted.reset_index()
    accepted = accepted.rename(columns={"": "id_patients", "created_at_assessments": 'meets_criteria'})
    # accepted.columns.values[0] = 'id_patients'
    print(accepted.columns)

    # Tidy up
    # accepted_df.reset_index(drop=True, inplace=True)
    # print keys
    return accepted

# Print basic summary info for a dataframe
def print_summary(dataframe,
                  df_name: str,
                  start_date=datetime.date(2022, 1, 1),
                  ):

    unique_patients = dataframe['id_patients'].nunique()
    # unique_assessments = dataframe.groupby('id_patients')['created_at_assessments'].nunique().sum()
    unique_assessments = dataframe.groupby('id_patients')['created_at_assessments'].nunique()

    tests_in_period = dataframe[dataframe['date_test'] >= start_date]
    # remove sum
    # total_tests = tests_in_period.groupby('id_patients')['date_test'].nunique().sum
    total_tests = tests_in_period.groupby('id_patients')['date_test'].nunique()

    print('Summary for', df_name, ':')
    print('Patients:', unique_patients)
    print('Assessments:', unique_assessments)
    print('Tests:', total_tests)

# Return assessments, tests and patients either side of cutoff date
def print_before_after_summary(dataframe,
                               start_date= datetime.date(2022,1,1),
                               policy_change_date = datetime.date(2022,4,1)):

    # Tests while free
    pat_tests_free = dataframe[dataframe['date_test'] < policy_change_date]
    pat_tests_while_free = pat_tests_free[pat_tests_free['date_test'] >= start_date]
    total_tests_while_free = pat_tests_while_free.groupby('id_patients')['date_test'].nunique().sum()
    print('Tests before policy change:', total_tests_while_free)

    # Tests after free testing ends
    pat_tests_after_free = dataframe[dataframe['date_test'] >= policy_change_date]
    total_tests_after_free = pat_tests_after_free.groupby('id_patients')['date_test'].nunique().sum()
    print('Tests after policy change:', total_tests_after_free)

    # Assessments while free
    assessments_while_free = dataframe[dataframe['created_at_assessments'] < policy_change_date]
    total_assessments_while_free = assessments_while_free.groupby('id_patients')[
        'created_at_assessments'].nunique().sum()
    print('Assessments before policy change:', total_assessments_while_free)

    # Assessments after free testing ends
    assessments_after_free = dataframe[dataframe['created_at_assessments'] >= policy_change_date]
    total_assessments_after_free = assessments_after_free.groupby('id_patients')[
        'created_at_assessments'].nunique().sum()
    print('Assessments after policy change:', total_assessments_after_free)

    patients_assessing_while_free = assessments_while_free['id_patients'].nunique()
    print('Patients providing assessments while free:', patients_assessing_while_free)
    patients_assessing_after_free = assessments_after_free['id_patients'].nunique()
    print('Patients providing assessments after free:', patients_assessing_after_free)

def return_weekly_average(series,
                          start_date: datetime,
                          end_date: datetime):
    # Calculate total days and weeks
    total_days = end_date - start_date
    weeks = (total_days.days / 7)
    # Get total tests per patient
    total_tests = len(series)
    # Return mean average
    mean_tests = total_tests / weeks
    # print('days:', total_days, 'weeks', weeks, 'tests', total_tests, 'mean tests', mean_tests)
    return mean_tests

# Function looks at tests per each week in time period, then returns median value
def return_weekly_median(series,
                         start_date: datetime,
                         end_date: datetime,
                         period=7):

    # convert period to datetime delta, calculate periods
    period = datetime.timedelta(days=period)
    total_days = end_date - start_date
    full_periods = ((end_date - start_date) // period)
    last_period = total_days % period
    # set running date
    running_date = start_date
    weekly_totals = []
# loop over the full periods, calculating assessment total in each window
    for i in range(full_periods):
        window_end = running_date + period
        total_in_window = sum(map(lambda x: x >= running_date and x < window_end, series))
        weekly_totals.append(total_in_window)
        running_date += period
  # add weighted tests for the remainder period
    if last_period.days != 0:
        total_in_last_window = (sum(map(lambda x: x >= (end_date - last_period) and x < end_date, series)))
        # print(end_date - last_period)
        # print('last period score:', total_in_last_window * period.days/last_period.days)
        weekly_totals.append(total_in_last_window * (period.days/last_period.days))

    # print('weekly totals:', weekly_totals)
    median = stats.median(weekly_totals)
    return median

def get_means_per_patient(dataframe,
                          start_date: datetime,
                          end_date: datetime):
    # Group dataframe by patients
    grouped = dataframe.groupby('id_patients')
    # Apply the averaging function to get an array of
    weekly_averages = grouped['date_test'].unique().apply(lambda x: return_weekly_average(x, start_date, end_date))
    # Give mean for all patients
    mean_of_all_patients = weekly_averages.mean()
    # Print and return the series, and overall mean
    return weekly_averages, mean_of_all_patients

def get_medians_per_patient(dataframe,
                            start_date: datetime,
                            end_date: datetime):
    # Group dataframe by patients
    grouped = dataframe.groupby('id_patients')
    # Apply the averaging function to get an array of
    weekly_medians = grouped['date_test'].unique().apply(lambda x: return_weekly_median(x, start_date, end_date))
    # Give mean for all patients
    median_of_all_patients = weekly_medians.median()
    # Print and return the series, and overall mean
    return weekly_medians, median_of_all_patients


def compare_freq_around_policy_change(dataframe,
                                      start_date: datetime,
                                      end_date: datetime,
                                      policy_change_date: datetime):
    """Returns a dataframe of unique patient IDs, with the mean
    number of tests taken before and after a policy change"""
    # Filter dataframe for tests before policy change
    tests_after_start_date = dataframe[dataframe['date_test'] >= start_date]
    tests_before_change = tests_after_start_date[tests_after_start_date['date_test'] < policy_change_date]
    # Get averages
    before_weekly_averages, before_average = get_means_per_patient(tests_before_change,
                                                                   start_date=start_date,
                                                                   end_date=policy_change_date)

    # Filter dataframe for tests before policy change
    tests_after_policy_change = dataframe[dataframe['date_test'] >= policy_change_date]
    tests_after_change = tests_after_policy_change[tests_after_policy_change['date_test'] < end_date]
    # Get averages
    after_weekly_averages, after_average = get_means_per_patient(tests_after_change,
                                                                 start_date=policy_change_date,
                                                                 end_date=end_date)

    # Create a new dataframe combining before and after averages
    df = pd.DataFrame({'mean_before': before_weekly_averages,
                       'mean_after': after_weekly_averages
                       })

    # Print summary
    # print('Mean before policy change:', before_average)
    # print('Mean after policy change:', after_average)

    return df

def compare_median_freq_around_policy_change(dataframe,
                                      start_date: datetime,
                                      end_date: datetime,
                                      policy_change_date: datetime):

    # Filter dataframe for tests before policy change
    tests_after_start_date = dataframe[dataframe['date_test'] >= start_date]
    tests_before_change = tests_after_start_date[tests_after_start_date['date_test'] < policy_change_date]
    # Get averages
    before_weekly_medians, before_median = get_medians_per_patient(tests_before_change,
                                                                   start_date=start_date,
                                                                   end_date=policy_change_date)

    # Filter dataframe for tests before policy change
    tests_after_policy_change = dataframe[dataframe['date_test'] >= policy_change_date]
    tests_after_change = tests_after_policy_change[tests_after_policy_change['date_test'] < end_date]
    # Get averages
    after_weekly_medians, after_median = get_medians_per_patient(tests_after_change,
                                                                 start_date=policy_change_date,
                                                                 end_date=end_date)

    # Create a new dataframe combining before and after averages
    df = pd.DataFrame({'median_before': before_weekly_medians,
                       'median_after': after_weekly_medians
                       })

    # Print summary
    # print('Median before policy change:', before_median)
    # print('Median after policy change:', after_median)

    return df

def return_df_assessments_tests_daily(dataframe):
    """Function to take a dataframe and return a new one
    which gives the total number of unique assessments and tests
    for every day"""

    # Get all assessments by day
    # Group unique assessments per patient
    unique_assessments = dataframe.groupby('id_patients')['created_at_assessments'].unique()
    # Convert to dataframe
    unique_assessments = pd.DataFrame(unique_assessments)
    # Create dataframe with one assessment per line
    b = unique_assessments.explode('created_at_assessments')
    # Count assessments per date
    c = b['created_at_assessments'].value_counts()
    # Convert to dataframe and reset index, colmn names
    c = pd.DataFrame(c)
    d = c.reset_index()
    e = d.rename(columns={"index": "date"})
    # Sort by date
    total_assessments_by_date = e.sort_values(by='date')


    # Get all tests by day
    # Group unique tests per patient
    unique_tests = dataframe.groupby('id_patients')['date_test'].unique()
    # Convert to dataframe
    unique_tests = pd.DataFrame(unique_tests)
    # Create dataframe with one assessment per line
    g = unique_tests.explode('date_test')
    # Count assessments per date
    h = g['date_test'].value_counts()
    # Convert to dataframe and reset index, column names
    i = pd.DataFrame(h)
    j = i.reset_index()
    k = j.rename(columns={"index": "date"})
    # Sort by date
    total_tests_by_date = k.sort_values(by='date')

    # Merge the dataframes
    new_df = total_assessments_by_date.merge(total_tests_by_date, how='inner', on='date')
    new_df['ratio'] = new_df['date_test'] / new_df['created_at_assessments']

    return new_df

def plot_daily_assess_tests(dataframe, country="Country Name"):
    """ Return a plot of assessments and tests over time"""

    # define colors to use
    col1 = 'steelblue'
    col2 = 'red'
    # define subplots
    fig, ax = plt.subplots()
    # add first line to plot
    ax.plot(dataframe.date, dataframe.created_at_assessments, color=col1, linewidth=2, label='Assessments')
    # add x-axis label
    ax.set_xlabel('Date', fontsize=14)
    # add y-axis label
    ax.set_ylabel('Assessments', fontsize=12)
    plt.legend(['Assessments'], loc="lower left")
    # define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()
    # add second line to plot
    ax2.plot(dataframe.date, dataframe.date_test, color=col2, linewidth=2, label='Tests')
    # add second y-axis label
    ax2.set_ylabel('Tests', fontsize=12)
    # Add title
    plt.title('Assessments & tests in {}'.format(country))
    ax.tick_params(axis='x', labelrotation=30)
    plt.legend(['Tests'], loc="upper right")

    return fig

# def plot_test_to_assess_ratio(dataframe):
#     """Function to plot a graph comparing the ratio of
#     tests to assessments between the different countries"""



def demographic_info(dataframe):
                     #start_date: datetime,
                     #end_date: datetime):
    # age distribution
    new_df = dataframe
    new_df['age_2022_start'] = 2021 - dataframe['year_of_birth']
    counts = new_df.groupby('id_patients')['age_2022_start'].unique()
    counts = pd.DataFrame(counts)
    counts = counts['age_2022_start'].value_counts()
    counts = counts.reset_index()
    counts_df = counts.rename(columns={"index": "age", "age_2022_start": "total"})
    counts_df['age'] = counts_df['age'].apply(lambda x: x[0])

    # does chemotherapy
    new_df_1 = dataframe
    chemo = new_df_1.groupby('id_patients')['does_chemotherapy'].unique()
    chemo = pd.DataFrame(chemo)
    chemo = chemo['does_chemotherapy'].value_counts()
    chemo = chemo.reset_index()
    chemo_df = chemo.rename(columns={"index": "chemotherapy_status", "does_chemotherapy": "total"})
    # chemo_df = chemo_df['chemotherapy_status'].replace({'[0]': "not provided"})

    # % healthcare workers
    # new_df_2 = dataframe
    # contact = new_df_2.groupby('id_patients')['contact_health_worker'].unique()
    # contact = pd.DataFrame(contact)
    # contact = contact['contact_health_worker'].value_counts()
    # contact = contact.reset_index()
    # contact_df = contact.rename(columns={"index": "contact_worker_status", "contact_health_worker": "total"})
    contact_df = "Needs updated data"

    # asthmatics
    new_df_3 = dataframe
    asthmatics = new_df_3.groupby('id_patients')['has_asthma'].unique()
    asthmatics = pd.DataFrame(asthmatics)
    asthmatics = asthmatics['has_asthma'].value_counts()
    asthmatics = asthmatics.reset_index()
    asthmatics_df = asthmatics.rename(columns={"index": "asthma_status", "has_asthma": "total"})

    # BMI status

    new_df_4 = dataframe
    bmi = new_df_4.groupby('id_patients')['bmi'].unique()
    bmi = pd.DataFrame(bmi)
    bmi = bmi['bmi'].value_counts(bins=[0, 18.5, 24.999999, 29.999999, 50])
    bmi = bmi.reset_index()
    bmi_df = bmi.rename(columns={"index": "bmi range", "bmi": "total"})

    # gender
    new_df_5 = dataframe
    gender = new_df_5.groupby('id_patients')['gender'].unique()
    gender = pd.DataFrame(gender)
    gender = gender['gender'].value_counts()
    gender = gender.reset_index()
    gender_df = gender.rename(columns={"index": "gender_values", "gender": "total"})

    return counts_df, chemo_df, contact_df, asthmatics_df, bmi_df, gender_df


def lower_stable_higher(x):
    """ Returns value corresponding to whether tests after was higher than
    testing before: 0 lower, 1 same, 2 higher"""

    if x < 0:
        return 0
    elif x == 0:
        return 1
    else:
        return 2


def compare_tests_around_policy_change(dataframe,
                                       start_date: datetime,
                                       end_date: datetime,
                                       policy_change_date: datetime):
    """Returns a dataframe of unique patient IDs, with the ratio
    of tests taken after a policy change compared to before"""

    # Filter dataframe for tests before policy change
    tests_after_start_date = dataframe[dataframe['date_test'] >= start_date]
    tests_before_change = tests_after_start_date[tests_after_start_date['date_test'] < policy_change_date]
    # Get total tests
    # Group dataframe by patients
    unique_tests_before = tests_before_change.groupby('id_patients')['date_test'].unique().apply(lambda x: len(x))
    unique_tests_before = pd.DataFrame(unique_tests_before)

    # Filter dataframe for tests before policy change
    tests_after_policy_change = dataframe[dataframe['date_test'] >= policy_change_date]
    tests_after_change = tests_after_policy_change[tests_after_policy_change['date_test'] < end_date]
    # Get total tests
    unique_tests_after = tests_after_change.groupby('id_patients')['date_test'].unique().apply(lambda x: len(x))
    unique_tests_after = pd.DataFrame(unique_tests_after)

    # Merge dataframes to give per patient totals
    df = unique_tests_before.merge(unique_tests_after, how='outer', on="id_patients")

    # Tidy up
    df = df.fillna(0)
    df = df.reset_index()
    df = df.rename(columns={"date_test_x": "total_before_policy_change", "date_test_y": "total_after_policy_change"})
    df['after_minus_before'] = df['total_after_policy_change'] - df['total_before_policy_change']
    df['lower_stable_higher'] = df['after_minus_before'].apply(lambda x: lower_stable_higher(x))
    df = df.drop(columns=['after_minus_before'])

    return df

def give_country_code(x):
  y = x[2:-1]
  z = y[0]
  if z == 'W':
    return 2
  elif z == 'S':
    return 1
  elif z == 'E':
    return 0
  elif z == '9':
    return 3
  else:
    return 4

def return_bmi_category(x):
  if x < 18.5:
    return 0
  elif 18.5 <= x < 25:
    return 1
  if 25 <= x < 30:
    return 2
  else:
    return 3

def convert_boolean(x):
  if x == False:
    return int(0)
  elif x == True:
    return int(1)

def features_dataframe(dataframe):
    """Will take a dataframe and extract values for each patient to feed into
    a model"""

    # Convert district codes to country
    # Retrieve unique patient IDs and country codes
    location = dataframe.groupby('id_patients')['lsoa11cd_x'].unique()
    location = pd.DataFrame(location)
    location['country'] = location['lsoa11cd_x'].apply(lambda x: give_country_code(x[0]))
    location = location.reset_index()
    location = location.drop(columns=['lsoa11cd_x'])

    # Add age for the patients
    age = dataframe.groupby('id_patients')['year_of_birth'].unique()
    age = pd.DataFrame(age)
    age['age_2022_start'] = 2021 - age['year_of_birth']
    age = age.reset_index()
    age['age'] = age['age_2022_start'].apply(lambda x: x[0])
    age = age.drop(columns=['year_of_birth', 'age_2022_start'])
    df = location.merge(age, how="inner", on='id_patients')

    # Add contact worker status
    health_worker = dataframe.groupby('id_patients')['contact_health_worker'].unique()
    health_worker = pd.DataFrame(health_worker)
    health_worker['health_worker_status'] = health_worker['contact_health_worker']#.apply(lambda x: convert_boolean(x))
    health_worker = health_worker.reset_index()
    health_worker = health_worker.drop(columns=['contact_health_worker'])
    df = df.merge(health_worker, how="inner", on='id_patients')

    # Add gender
    gender = dataframe.groupby('id_patients')['gender'].unique()
    gender = pd.DataFrame(gender)
    gender = gender.reset_index()
    gender['gender'] = gender['gender'].apply(lambda x: x[0] - 1) # -1 moves from CSS coding... so that female:0, male:1
    df = df.merge(gender, how="inner", on='id_patients')
    # Remove patients with gender: "not set, intersex, prefer not to say"
    genders_to_exclude = [-1, 2, 3]
    df = df[df.gender.isin(genders_to_exclude) == False]

    # Add BMI
    bmi = dataframe.groupby('id_patients')['bmi'].unique()
    bmi = pd.DataFrame(bmi)
    bmi = bmi.reset_index()
    bmi['bmi_range'] = bmi['bmi'].apply(lambda x: return_bmi_category(x[0]))
    bmi = bmi.drop(columns=['bmi'])
    df = df.merge(bmi, how="inner", on='id_patients')

    # Add preconditions
    preconditions = dataframe[
        ['id_patients', 'has_cancer', 'has_asthma', 'does_chemotherapy', 'has_heart_disease', 'has_lung_disease']]

    conditions = [preconditions.has_cancer == 2,
                  preconditions.has_asthma == 2,
                  preconditions.does_chemotherapy == 2,
                  preconditions.has_heart_disease == 2,
                  preconditions.has_lung_disease == 2]

    value = [1, 1, 1, 1, 1]

    preconditions['has_preconditions'] = np.select(conditions, value)
    preconditions_sum = pd.merge(df, preconditions[['id_patients', 'has_preconditions']], on='id_patients', how='left')
    preconditions_sum = preconditions_sum.groupby('id_patients')['has_preconditions'].unique()
    preconditions_sum = preconditions_sum.reset_index()
    preconditions_sum['preconditions_status'] = preconditions_sum['has_preconditions'].apply(
        lambda x: 1 if max(x) > 0 else 0)
    preconditions_sum = preconditions_sum.drop(columns='has_preconditions')

    df = df.merge(preconditions_sum, how="inner", on='id_patients')
    print(df)

    return df


def EvaluatePerformance(model, X, y, modeltitle:str, test_or_train:str):
    "A function to evaluate the performance of a model on the training data, taking the model,"
    "and a title for the model as arguments, and printing cross validated accuracy"
    "sensitivity, specificity and mean recall"

    print('{} performance'.format(modeltitle) + ' ' + 'on {} data:'.format(test_or_train))

    scores = cross_val_score(model, X, y, cv=5)
    print('Accuracy: ', round(scores.mean(), 2))

    y_pred = cross_val_predict(model, X, y)

    sensitivity = recall_score(y, y_pred, pos_label=1)
    print('Sensitivity: ', round(sensitivity, 2))

    specificity = recall_score(y, y_pred, pos_label=0)
    print('Specificity: ', round(specificity, 2))

    mean_recall = recall_score(y, y_pred, average='macro')
    print('Mean recall: ', round(mean_recall, 2))


def write_df_to_csv(dataframe, filename:str):
    columns = dataframe.columns
    values = dataframe.values

    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(values)

def stratified_sample_requirements(dataframe, train_percentage=80, test_percentage=20):
  """Given a dataframe, and a specified split for the test and train folds,
  prints the ideal number of patiens of each category to be sampled in a test
  populatio"""

  dataframe_columns = dataframe.keys()
  for column in dataframe_columns:
    print(column)
    x = dataframe['{}'.format(column)].value_counts()
    y = dataframe['{}'.format(column)].value_counts() // ((train_percentage + test_percentage) / (test_percentage))
    print('Total:', x)
    print('20%:', y)

def categorise_age(x):
  """Places a given age (in years) into predefined categories and returns
  category value"""
  if x < 25:
    return 0
  elif 25 <= x < 40:
    return 1
  elif 40 <= x < 55:
    return 2
  elif 55 <= x < 70:
    return 3
  else:
    return 4

def stratify_ages(dataframe):
  """Returns a dataframe with an added column containing age stratification"""
  dataframe['age_category'] = dataframe['age'].apply(lambda x: categorise_age(x))
  return dataframe


def return_acceptable_range(desired_total: int,
                            min_threshold=0.99,
                            max_threshold=1.01):
    """For an ideal number of patients in a test population, this function
    returns an acceptable range, within specified min and max thresholds"""

    min = int(desired_total * min_threshold)
    max = int(desired_total * max_threshold)
    return range(min, max)

def check_in_range(desired, range):
    """Checks whether a value is in a range and returns boolean"""
    if desired in range:
        return True
    else:
        return False

def create_stratified_test_sample(dataframe,
                                  required_ages: dict,
                                  required_genders: dict,
                                  required_hw: dict,
                                  required_precs: dict,
                                  required_bmi: dict,
                                  required_target: dict
                                  ):

  # Create the subpopulations
  age_0 = dataframe[(dataframe.age_category == 0)]
  age_1 = dataframe[(dataframe.age_category == 1)]
  age_2 = dataframe[(dataframe.age_category == 2)]
  age_3 = dataframe[(dataframe.age_category == 3)]
  age_4 = dataframe[(dataframe.age_category == 4)]

  # Create the test sample for each category & concatenate
  age_0_test = age_0.sample(required_ages[0], replace=False)
  age_1_test = age_1.sample(required_ages[1], replace=False)
  age_2_test = age_2.sample(required_ages[2], replace=False)
  age_3_test = age_3.sample(required_ages[3], replace=False)
  age_4_test = age_4.sample(required_ages[4], replace=False)

  test_sample = pd.concat([pd.DataFrame(age_0_test),
                           pd.DataFrame(age_1_test),
                           pd.DataFrame(age_2_test),
                           pd.DataFrame(age_3_test),
                           pd.DataFrame(age_4_test),
                           ], axis=0)

  print(test_sample.head(20))
  test_sample = pd.DataFrame(test_sample)
  print(type(test_sample))

  # get acceptable ranges
  gender_range = return_acceptable_range(required_genders[0])
  contact_hw_range = return_acceptable_range(required_hw[0])
  precs_range = return_acceptable_range(required_precs[0])
  underweight_range = return_acceptable_range(required_bmi[0])
  healthy_range = return_acceptable_range(required_bmi[1])
  overweight_range = return_acceptable_range(required_bmi[2])
  target_range = return_acceptable_range(required_target[0])

  # get values in test sample
  gender_totals_in_test = test_sample['gender'].value_counts()
  print('gender totals')
  print(gender_totals_in_test)
  females_in_test = gender_totals_in_test[0]

  contact_hw_in_test = test_sample['health_worker_status'].value_counts()
  print('contact totals')
  print(contact_hw_in_test)
  non_hw_in_test = contact_hw_in_test[0]

  preconditions_in_test = test_sample['preconditions_status'].value_counts()
  print('preconditions')
  print(preconditions_in_test)
  no_precs_in_test = preconditions_in_test[0]

  bmi_in_test = test_sample['bmi_range'].value_counts()
  print('bmi')
  print(bmi_in_test)
  underweight_in_test = bmi_in_test[0]
  healthy_in_test = bmi_in_test[1]
  overweight_in_test = bmi_in_test[2]

  target_in_test = test_sample['lower_higher'].value_counts()
  print('target')
  print(target_in_test)
  lower_in_test = target_in_test[0]

  # check values in the sample are in acceptable ranges
  gender = check_in_range(females_in_test, gender_range)
  contact_hw = check_in_range(non_hw_in_test, contact_hw_range)
  preconditions = check_in_range(no_precs_in_test, precs_range)
  underweight = check_in_range(underweight_in_test, underweight_range)
  healthy = check_in_range(healthy_in_test, healthy_range)
  overweight = check_in_range(overweight_in_test, overweight_range)
  target = check_in_range(lower_in_test, target_range)
  print(gender, contact_hw, preconditions, underweight, healthy, overweight, target)

  if gender and contact_hw and preconditions and underweight and healthy and overweight and target == True:
      test_sample = test_sample.reset_index(drop=True)
      print('len test:', len(test_sample))
      write_df_to_csv(test_sample, filename="stratified_test_sample.csv")
      print('test sample saved')

      train_sample = pd.concat([test_sample, dataframe]).drop_duplicates(keep=False)
      print('len train:', len(train_sample))
      train_sample = train_sample.reset_index(drop=True)
      write_df_to_csv(train_sample, filename="stratified_train_sample.csv")
      print('train sample saved')

      return 0



  else:
      print('no good, resampling...')
      create_stratified_test_sample(dataframe, required_ages, required_genders,
                                    required_hw, required_precs, required_bmi, required_target)

