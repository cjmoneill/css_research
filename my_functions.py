import pandas as pd
import datetime as dt
import numpy as np
import datetime

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
    # Create a dataframe with unique patient IDs
    unique_patients = dataframe['id_patients'].unique()
    # Add the accepted column
    accepted_df = pd.DataFrame({'id_patients': unique_patients, 'meets_criteria': accepted})
    # Tidy up
    accepted_df.reset_index(drop=True, inplace=True)
    return accepted_df

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