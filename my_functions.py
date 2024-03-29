import pandas as pd
import datetime as dt
import numpy as np
import datetime
import statistics as stats
import matplotlib.pyplot as plt
import csv
import itertools
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


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
    # new_df[series] = dataframe[series].replace('0', '2001, 1, 1')
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
    ax.set_xlabel('Date', fontsize=12)
    # add y-axis label
    ax.set_ylabel('Assessments', fontsize=12)
    # ax.axis([datetime.date(2022,2,1), datetime.date(2022,5,22), 0, 200000])
    plt.legend(['Assessments'], loc="lower left")
    # define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()
    # add second line to plot
    ax2.plot(dataframe.date, dataframe.date_test, color=col2, linewidth=2, label='Tests')
    # add second y-axis label
    ax2.set_ylabel('Tests', fontsize=12)
    # Add title
    plt.title('CSS assessments & tests around policy change in {}'.format(country), fontsize=14)
    plt.axvline(x=datetime.date(2022,4,1), color='g', label='Policy change')
    ax.tick_params(axis='x', labelrotation=30)
    plt.subplots_adjust(bottom=0.4)
    plt.legend(['Tests'], loc="upper right")
    ax2.axis([datetime.date(2022,2,1), datetime.date(2022,5,22), 0, 4000])
    plt.tight_layout()

    return fig

def plot_pos_test_ratio(dataframe):
    """Function to plot a graph comparing the ratio of
    tests to assessments between the different countries"""

    # define colors to use
    col1 = 'steelblue'
    col2 = 'red'
    # define subplots
    fig, ax = plt.subplots()
    # add first line to plot
    ax.plot(dataframe.day, dataframe.ratio, color=col1, linewidth=2, label='Ratio of positive tests')
    # add x-axis label
    ax.set_xlabel('Date', fontsize=12)
    # add y-axis label
    ax.set_ylabel('Assessments', fontsize=12)
    # ax.axis([datetime.date(2022,2,1), datetime.date(2022,5,22), 0, 200000])
    plt.legend(['Assessments'], loc="lower left")
    # Add title
    plt.title('Ratio of positive tests recorded via CSS', fontsize=14)
    plt.axvline(x=datetime.date(2022, 4, 1), color='g', label='Policy change')
    ax.tick_params(axis='x', labelrotation=30)
    plt.subplots_adjust(bottom=0.4)
    plt.legend(['Ratio of positive tests'], loc="upper right")
    plt.tight_layout()

    return fig


def age_distribution(dataframe):

    new_df = dataframe
    new_df['age_2022_start'] = 2021 - dataframe['year_of_birth']
    new_df['age_category'] = new_df['age_2022_start'].apply(lambda x: categorise_age(x))
    counts = new_df.groupby('id_patients')['age_category'].unique()
    counts = pd.DataFrame(counts)
    counts = counts['age_category'].value_counts()
    counts = counts.reset_index()
    counts_df = counts.rename(columns={"index": "age_2022_start", "age_category": "total"})

    return counts_df

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
    new_df_2 = dataframe
    contact = new_df_2.groupby('id_patients')['contact_health_worker'].unique()
    contact = pd.DataFrame(contact)
    contact = contact['contact_health_worker'].value_counts()
    contact = contact.reset_index()
    contact_df = contact.rename(columns={"index": "contact_worker_status", "contact_health_worker": "total"})
    # contact_df = "Needs updated data"

    # asthmatics
    new_df_3 = dataframe
    asthmatics = new_df_3.groupby('id_patients')['has_asthma'].unique()
    asthmatics = pd.DataFrame(asthmatics)
    asthmatics = asthmatics['has_asthma'].value_counts()
    asthmatics = asthmatics.reset_index()
    asthmatics_df = asthmatics.rename(columns={"index": "asthma_status", "has_asthma": "total"})

    # cancer
    new_df_6 = dataframe
    cancer = new_df_6.groupby('id_patients')['has_cancer'].unique()
    cancer = pd.DataFrame(cancer)
    cancer = cancer['has_cancer'].value_counts()
    cancer = cancer.reset_index()
    cancer_df = cancer.rename(columns={"index": "cancer_status", "has_cancer": "total"})

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

    # heart disease
    new_df_7 = dataframe
    hd = new_df_7.groupby('id_patients')['has_heart_disease'].unique()
    hd = pd.DataFrame(hd)
    hd = hd['has_heart_disease'].value_counts()
    hd = hd.reset_index()
    hd_df = hd.rename(columns={"index": "heart_disease_values", 'has_heart_disease': "total"})

    # lung disease
    new_df_8 = dataframe
    ld = new_df_8.groupby('id_patients')['has_lung_disease'].unique()
    ld = pd.DataFrame(ld)
    ld = ld['has_lung_disease'].value_counts()
    ld = ld.reset_index()
    ld_df = ld.rename(columns={"index": "lung_disease_values", 'has_lung_disease': "total"})


    return counts_df, chemo_df, contact_df, asthmatics_df, bmi_df, gender_df, cancer_df, hd_df, ld_df


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

    # Filter dataframe for tests after policy change
    tests_after_policy_change = dataframe[dataframe['date_test'] >= policy_change_date]
    tests_after_change = tests_after_policy_change[tests_after_policy_change['date_test'] <= end_date]
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

def add_bmi_catgory_column(dataframe):
    new_df = dataframe
    new_df['bmi_category'] = dataframe['bmi'].apply(lambda x: return_bmi_category(x))
    return new_df

def convert_boolean(x):
  if x == False:
    return int(0)
  elif x == True:
    return int(1)

def convert_nan_to_zero(x):
    if x == '[nan]':
        return int(0)
    elif x == '[NaN]':
        return int(0)

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
    health_worker['health_worker_status'] = health_worker['contact_health_worker'].apply(lambda x: convert_boolean(x))
    health_worker['health_worker_status_updated'] = health_worker.health_worker_status.replace('nan', 0)

    # health_worker['health_worker_status'] = health_worker['contact_health_worker'].apply(lambda x: convert_nan_to_zero(x))
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

def EvaluatePerformance(model, X, y, modeltitle:str, matrix_title='Confusion matrix'):
    "A function to evaluate the performance of a model on the training data, taking the model,"
    "and a title for the model as arguments, and printing cross validated accuracy"
    "sensitivity, specificity and mean recall"

    print('{} performance'.format(modeltitle))

    y_pred = model.predict(X)

    accuracy = accuracy_score(y,y_pred)
    print('accuracy: ', round(accuracy, 2))

    sensitivity = recall_score(y, y_pred, pos_label=1)
    print('Sensitivity: ', round(sensitivity, 2))

    specificity = recall_score(y, y_pred, pos_label=0)
    print('Specificity: ', round(specificity, 2))

    mean_recall = recall_score(y, y_pred, average='macro')
    print('Mean recall: ', round(mean_recall, 2))

    matrix = confusion_matrix(y, y_pred, normalize='true')
    display_matrix = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                            display_labels=model.classes_,
                                            )
    display_matrix.plot()
    plt.title(matrix_title)
    plt.show()

def EvaluatePerformanceCV(model, X, y, modeltitle:str, matrix_title='Confusion matrix'):
    "A function to evaluate the performance of a model on the training data, taking the model,"
    "and a title for the model as arguments, and printing cross validated accuracy"
    "sensitivity, specificity and mean recall"

    print('{} performance'.format(modeltitle) + ' with cross validation')

    y_pred = cross_val_predict(model, X, y)

    accuracy = accuracy_score(y,y_pred)
    print('accuracy: ', round(accuracy, 2))

    sensitivity = recall_score(y, y_pred, pos_label=1)
    print('Sensitivity: ', round(sensitivity, 2))

    specificity = recall_score(y, y_pred, pos_label=0)
    print('Specificity: ', round(specificity, 2))

    mean_recall = recall_score(y, y_pred, average='macro')
    print('Mean recall: ', round(mean_recall, 2))

    matrix = confusion_matrix(y, y_pred, normalize='true')
    display_matrix = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                            display_labels=model.classes_)
    display_matrix.plot()
    plt.title(matrix_title)
    plt.show()


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

def stratify_ages(dataframe, key='age'):
  """Returns a dataframe with an added column containing age stratification"""
  dataframe['age_category'] = dataframe[key].apply(lambda x: categorise_age(x))
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
                                  required_target: dict,
                                  name_suffix='1'
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
      write_df_to_csv(test_sample, filename="stratified_test_sample_{}.csv".format(name_suffix))
      print('test sample saved')

      train_sample = pd.concat([test_sample, dataframe]).drop_duplicates(keep=False)
      print('len train:', len(train_sample))
      train_sample = train_sample.reset_index(drop=True)
      write_df_to_csv(train_sample, filename="stratified_train_sample_{}.csv".format(name_suffix))
      print('train sample saved')

      return 0


  else:
      print('no good, resampling...')
      create_stratified_test_sample(dataframe, required_ages, required_genders,
                                    required_hw, required_precs, required_bmi, required_target,
                                    name_suffix)


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):

    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

def plot_search_results(grid):
    """Plot search results when supplying a trained GridSearchCV object"""

    ## Results from grid search
    results = grid.cv_results_
    # print(results)
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    # means_train = results['mean_train_score']
    # stds_train = results['std_train_score']

    ## Get indexes of values per hyper-parameter
    masks_names = list(grid.best_params_.keys())
    # print('best params items:', grid.best_params_.items() )
    masks=[]
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params = grid.param_grid
    # print('mask names:', masks_names)
    # print('params:', params)
    # print('len params:', len(params))

    ## Plotting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_performance_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_performance_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        # y_2 = np.array(means_train[best_index])
        # e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='train')
        # ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()

def one_hot_encode(dataframe, *args, label: str):
    """A function which will one hot encode a  specific column and
     return a new dataframe which includes one
     Args: a list of labels which correspond to [0, 1, 2 etc.]
     Label: the key name for the column to be one hot encoded"""
    # One hot encode categorical columns where not binary
    # Replace numeric values with labels

    # Create labels dictionary
    labels_dict = {}
    for i, x in enumerate(*args):
        labels_dict[i] = x
    print(labels_dict)

    # Create new categorical column
    dataframe['{} categories'.format(label)] = dataframe['{}'.format(label)].replace(labels_dict)
    print(dataframe)

    # One hot encode
    one_hot_columns = pd.get_dummies(dataframe['{} categories'.format(label)])
    print(one_hot_columns)

    # Merge
    new_dataframe = pd.concat([dataframe, one_hot_columns], axis=1)

    # Drop old columns
    new_dataframe = new_dataframe.drop(columns=['{} categories'.format(label)])

    return new_dataframe

def return_df_of_totals_relative(dataframe,
                                 start_date: datetime,
                                 end_date: datetime,
                                 policy_change: datetime,
                                 period=7):
    # Get unique tests
    unique_tests_df = dataframe.groupby('id_patients')['date_test'].unique()
    unique_tests_df = pd.DataFrame(unique_tests_df)
    unique_tests_df = unique_tests_df.reset_index()

    # Calculate periods before & after
    period = datetime.timedelta(days=period)

    total_days_before = policy_change - start_date
    periods_before = total_days_before // period

    total_days_after = end_date - policy_change
    periods_after = total_days_after // period
    print('full periods before:', periods_before)
    print('full periods after:', periods_after)

    # Set running date & period label
    running_date = policy_change
    period_relative_to_change = -1

    # Create new dataframe to copy into
    tests_per_period = pd.DataFrame(columns=['id_patients', 'period_relative_to_change', 'tests_in_period'])

    # Loop over the periods before, calculate tests, add to new dataframe
    for index, row in unique_tests_df.iterrows():
        period_relative_to_change = -1
        running_date = policy_change

        for i in range(periods_before):
            window_begin = running_date - period
            total_in_window = sum(map(lambda x: x >= window_begin and x < running_date, row['date_test']))
            df = {'id_patients': row['id_patients'],
                  'period_relative_to_change': period_relative_to_change,
                  'tests_in_period': total_in_window}
            tests_per_period = tests_per_period.append(df, ignore_index=True)
            period_relative_to_change += -1
            running_date = window_begin

    running_date = policy_change
    period_relative_to_change = 0

    # Loop over the periods after, calculate tests, add to new dataframe
    for index, row in unique_tests_df.iterrows():
        period_relative_to_change = 0
        running_date = policy_change

        for i in range(periods_after):
            window_end = running_date + period
            total_in_window = sum(map(lambda x: x >= running_date and x < window_end, row['date_test']))
            df = {'id_patients': row['id_patients'],
                  'period_relative_to_change': period_relative_to_change,
                  'tests_in_period': total_in_window}
            tests_per_period = tests_per_period.append(df, ignore_index=True)
            period_relative_to_change += +1
            running_date = window_end

    # Tidy up to ordered by patients and test period
    tests_per_period = tests_per_period.sort_values(['id_patients', 'period_relative_to_change'],
                                                    ascending=[True, True], ignore_index=True)

    return tests_per_period


def periods_symptomatic(dataframe,
                        start_date: datetime,
                        end_date: datetime,
                        policy_change: datetime,
                        period=7):

    # create 'is symptomatic column'
    dataframe['symptomatic'] = dataframe['headache'] + dataframe['loss_of_smell'] + dataframe['persistent_cough'] + dataframe['sore_throat'] + dataframe['runny_nose'] + dataframe['fever'] + dataframe['shortness_of_breath']

    # drop non-symptomatic assessments
    dataframe = dataframe[dataframe['symptomatic'] > 0]

    # get unique days with symptomatic assessments per patient
    unique_sympt_days_df = dataframe.groupby('id_patients')['created_at_assessments'].unique()
    unique_sympt_days_df = pd.DataFrame(unique_sympt_days_df)
    unique_sympt_days_df = unique_sympt_days_df.reset_index()
    print(unique_sympt_days_df)

    # Calculate periods before & after
    period = datetime.timedelta(days=period)

    total_days_before = policy_change - start_date
    periods_before = total_days_before // period

    total_days_after = end_date - policy_change
    print('total days after:', total_days_after)
    periods_after = total_days_after // period
    print('full periods before:', periods_before)
    print('full periods after:', periods_after)

    # Set running date & period label
    running_date = policy_change
    period_relative_to_change = -1

    # Create new dataframe to copy into
    symptomatic_per_period = pd.DataFrame(
        columns=['id_patients', 'period_relative_to_change', 'symptomatic_in_period', 'days_symptomatic'])
    print(symptomatic_per_period)

    # Loop over the periods before, calculate days swmptomatic, add to new dataframe
    for index, row in unique_sympt_days_df.iterrows():
        period_relative_to_change = -1
        running_date = policy_change

        for i in range(periods_before):
            window_begin = running_date - period
            total_in_window = sum(map(lambda x: x >= window_begin and x < running_date, row['created_at_assessments']))
            df = {'id_patients': row['id_patients'],
                  'period_relative_to_change': period_relative_to_change,
                  'symptomatic_in_period': 1 if total_in_window > 0 else 0,
                  'days_symptomatic': total_in_window}
            symptomatic_per_period = symptomatic_per_period.append(df, ignore_index=True)
            period_relative_to_change += -1
            running_date = window_begin

        running_date = policy_change
        period_relative_to_change = 0

    # Loop over the periods after, calculate tests, add to new dataframe
    for index, row in unique_sympt_days_df.iterrows():
        period_relative_to_change = 0
        running_date = policy_change

        for i in range(periods_after):
            window_end = running_date + period
            total_in_window = sum(map(lambda x: x >= running_date and x < window_end, row['created_at_assessments']))
            df = {'id_patients': row['id_patients'],
                  'period_relative_to_change': period_relative_to_change,
                  'symptomatic_in_period': 1 if total_in_window > 0 else 0,
                  'days_symptomatic': total_in_window}
            symptomatic_per_period = symptomatic_per_period.append(df, ignore_index=True)
            period_relative_to_change += +1
            running_date = window_end

    # Tidy up to ored by patients and test period
    symptomatic_per_period = symptomatic_per_period.sort_values(['id_patients', 'period_relative_to_change'],
                                                                ascending=[True, True], ignore_index=True)

    return symptomatic_per_period


def return_df_of_assessments(dataframe,
                             start_date: datetime,
                             end_date: datetime,
                             policy_change: datetime,
                             period=7):
    # Get unique assessments
    unique_assess_df = dataframe.groupby('id_patients')['created_at_assessments'].unique()
    unique_assess_df = pd.DataFrame(unique_assess_df)
    unique_assess_df = unique_assess_df.reset_index()

    # Calculate periods before & after
    period = datetime.timedelta(days=period)

    total_days_before = policy_change - start_date
    periods_before = total_days_before // period

    total_days_after = end_date - policy_change
    print('total days after:', total_days_after)
    periods_after = total_days_after // period
    print('full periods before:', periods_before)
    print('full periods after:', periods_after)

    # Set running date & period label
    running_date = policy_change
    period_relative_to_change = -1

    # Create new dataframe to copy into
    assess_per_period = pd.DataFrame(columns=['id_patients', 'period_relative_to_change', 'assessments_in_period'])
    print(assess_per_period)

    # Loop over the periods before, calculate tests, add to new dataframe
    for index, row in unique_assess_df.iterrows():
        period_relative_to_change = -1
        running_date = policy_change

        for i in range(periods_before):
            window_begin = running_date - period
            total_in_window = sum(map(lambda x: x >= window_begin and x < running_date, row['created_at_assessments']))
            df = {'id_patients': row['id_patients'],
                  'period_relative_to_change': period_relative_to_change,
                  'assessments_in_period': total_in_window}
            assess_per_period = assess_per_period.append(df, ignore_index=True)
            period_relative_to_change += -1
            running_date = window_begin

    running_date = policy_change
    period_relative_to_change = 0

    # Loop over the periods after, calculate tests, add to new dataframe
    for index, row in unique_assess_df.iterrows():
        period_relative_to_change = 0
        running_date = policy_change

        for i in range(periods_after):
            window_end = running_date + period
            total_in_window = sum(map(lambda x: x >= running_date and x < window_end, row['created_at_assessments']))
            df = {'id_patients': row['id_patients'],
                  'period_relative_to_change': period_relative_to_change,
                  'assessments_in_period': total_in_window}
            assess_per_period = assess_per_period.append(df, ignore_index=True)
            period_relative_to_change += +1
            running_date = window_end

    # Tidy up to ored by patients and test period
    assess_per_period = assess_per_period.sort_values(['id_patients', 'period_relative_to_change'],
                                                      ascending=[True, True], ignore_index=True)

    return assess_per_period


def symptoms_in_period(dataframe,
                       start_date: datetime,
                       end_date: datetime,
                       policy_change: datetime,
                       period=7):
    # create new dataframe to copy into
    symptoms_per_period = pd.DataFrame(
        columns=['id_patients', 'period_relative_to_change', 'symptom_set', 'unique_symptoms_total',
                 'total_symptom_days'])

    # create symptoms list column
    dataframe['headache_a'] = dataframe['headache'].apply(lambda x: ['headache'] if x > 0 else [])
    dataframe['loss_of_smell_a'] = dataframe['loss_of_smell'].apply(lambda x: ['loss_of_smell'] if x > 0 else [])
    dataframe['persistent_cough_a'] = dataframe['persistent_cough'].apply(lambda x: ['persistent_cough'] if x > 0 else [])
    dataframe['sore_throat_a'] = dataframe['sore_throat'].apply(lambda x: ['sore_throat'] if x > 0 else [])
    dataframe['runny_nose_a'] = dataframe['runny_nose'].apply(lambda x: ['runny_nose'] if x > 0 else [])
    dataframe['fever_a'] = dataframe['fever'].apply(lambda x: ['fever'] if x > 0 else [])
    dataframe['shortness_of_breath_a'] = dataframe['shortness_of_breath'].apply(lambda x: ['shortness_of_breath'] if x > 0 else [])

    dataframe['symptoms_list'] = dataframe['headache_a'] + dataframe['loss_of_smell_a'] + dataframe['persistent_cough_a'] + dataframe['sore_throat_a'] + dataframe['runny_nose_a'] + dataframe['fever_a'] + dataframe['shortness_of_breath_a']
    dataframe['symptoms_tuple'] = dataframe['symptoms_list'].apply(lambda x: tuple(x))

    # Calculate periods before & after
    period = datetime.timedelta(days=period)

    total_days_before = policy_change - start_date
    periods_before = total_days_before // period

    total_days_after = end_date - policy_change
    print('total days after:', total_days_after)
    periods_after = total_days_after // period
    print('full periods before:', periods_before)
    print('full periods after:', periods_after)

    # Set running date & period label
    running_date = policy_change
    period_relative_to_change = -1

    # Loop over the weeks before policy change
    for week in range(periods_before):
        window_begin = running_date - period

        working_dataframe = dataframe[dataframe['created_at_assessments'] < running_date]
        working_dataframe = working_dataframe[working_dataframe['created_at_assessments'] >= window_begin]

        # get  all the unique symptoms
        unique_patient_symptoms = working_dataframe.groupby('id_patients')['symptoms_tuple'].unique()
        unique_patient_symptoms = pd.DataFrame(unique_patient_symptoms)
        unique_patient_symptoms = unique_patient_symptoms.reset_index()

        # unpack all of them
        unique_patient_symptoms['unpacked'] = unique_patient_symptoms['symptoms_tuple'].apply(
            lambda x: list(itertools.chain.from_iterable(x)))

        # Create column for unique symptoms in period
        unique_patient_symptoms['symptom_set'] = unique_patient_symptoms['unpacked'].apply(lambda x: set(x))

        # Add total symptom days and unique symptoms to the dataframe
        unique_patient_symptoms['unique_symptoms_total'] = unique_patient_symptoms['symptom_set'].apply(
            lambda x: len(x))
        unique_patient_symptoms['total_symptom_days'] = unique_patient_symptoms['unpacked'].apply(lambda x: len(x))

        # Tidy up
        unique_patient_symptoms = unique_patient_symptoms.drop(columns=['symptoms_tuple', 'unpacked'])

        # Add period relative to change
        unique_patient_symptoms['period_relative_to_change'] = period_relative_to_change
        print('unique_patient_symptoms:', unique_patient_symptoms)

        # Add to main dataframe & update running dates
        symptoms_per_period = symptoms_per_period.append(unique_patient_symptoms, ignore_index=True)
        period_relative_to_change += -1
        running_date = window_begin

    # Update running date & period label
    running_date = policy_change
    period_relative_to_change = 0

    # Loop over the weeks before policy change
    for week in range(periods_after):
        window_end = running_date + period

        working_dataframe = dataframe[dataframe['created_at_assessments'] < window_end]
        working_dataframe = working_dataframe[working_dataframe['created_at_assessments'] >= running_date]

        # get  all the unique symptoms
        unique_patient_symptoms = working_dataframe.groupby('id_patients')['symptoms_tuple'].unique()
        unique_patient_symptoms = pd.DataFrame(unique_patient_symptoms)
        unique_patient_symptoms = unique_patient_symptoms.reset_index()

        # unpack all of them
        unique_patient_symptoms['unpacked'] = unique_patient_symptoms['symptoms_tuple'].apply(
            lambda x: list(itertools.chain.from_iterable(x)))

        # Create column for unique symptoms in period
        unique_patient_symptoms['symptom_set'] = unique_patient_symptoms['unpacked'].apply(lambda x: set(x))

        # Add total symptom days and unique symptoms to the dataframe
        unique_patient_symptoms['unique_symptoms_total'] = unique_patient_symptoms['symptom_set'].apply(
            lambda x: len(x))
        unique_patient_symptoms['total_symptom_days'] = unique_patient_symptoms['unpacked'].apply(lambda x: len(x))

        # Tidy up
        unique_patient_symptoms = unique_patient_symptoms.drop(columns=['symptoms_tuple', 'unpacked'])

        # Add period relative to change
        unique_patient_symptoms['period_relative_to_change'] = period_relative_to_change
        print('unique_patient_symptoms:', unique_patient_symptoms)

        # Add to main dataframe & update running dates
        symptoms_per_period = symptoms_per_period.append(unique_patient_symptoms, ignore_index=True)
        period_relative_to_change += 1
        running_date = window_end

    # Tidy up to be ordered by patients and test period
    symptoms_per_period = symptoms_per_period.sort_values(['id_patients', 'period_relative_to_change'],
                                                          ascending=[True, True], ignore_index=True)

    return symptoms_per_period


def skeleton_features_df(dataframe,
                         start_date: datetime,
                         end_date: datetime,
                         policy_change: datetime,
                         period=7):
    """A function to create a skeleton dataframe for all patients, and all periods before
    and after a policy change, to be populated with tests, assessments and symptom data"""

    # Calculate periods before & after
    period = datetime.timedelta(days=period)
    total_days_before = policy_change - start_date
    periods_before = total_days_before // period
    total_days_after = end_date - policy_change
    periods_after = total_days_after // period
    print('full periods before:', periods_before)
    print('full periods after:', periods_after)

    # Get unique patients
    unique_patients = dataframe['id_patients'].unique()
    print(unique_patients)

    # Create dataframe
    features_skeleton = pd.DataFrame(columns=['id_patients', 'period_relative_to_change'])

    # period_relative_to_change = -1
    # Create rows for each period before
    for patient in unique_patients:
        period_relative_to_change = -1
        for i in range(periods_before):
            df = {'id_patients': patient,
                  'period_relative_to_change': period_relative_to_change}
            features_skeleton = features_skeleton.append(df, ignore_index=True)
            period_relative_to_change += -1

    for patient in unique_patients:
        period_relative_to_change = 0
        for i in range(periods_before):
            df = {'id_patients': patient,
                  'period_relative_to_change': period_relative_to_change}
            features_skeleton = features_skeleton.append(df, ignore_index=True)
            period_relative_to_change += 1

    # Tidy up to be ordered by patients and test period
    features_skeleton = features_skeleton.sort_values(['id_patients', 'period_relative_to_change'],
                                                      ascending=[True, True], ignore_index=True)

    return features_skeleton


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()