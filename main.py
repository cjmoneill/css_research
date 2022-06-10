import datetime
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
import datetime as dt

import my_functions as mf
from my_functions import exclude_invalid_bmi
from my_functions import convert_df_dates
from my_functions import convert_df_date_series
from my_functions import assessment_frequency_criteria
from my_functions import return_dataframe_of_accepted_patients
from my_functions import print_before_after_summary

print('imports complete')


def main(path, start_date: datetime):
    print('running')
    print('Period starts:', start_date)

    # Import the tests csv file into a dataframe
    pat_tests = pd.read_csv(path)

    # Print some basic info (need more!)
    print('Total assessments_tests:', len(pat_tests))
    unique_patients_initial = pat_tests['id_patients'].nunique()
    unique_patients_assessments = pat_tests.groupby('id_patients')['created_at_assessments'].nunique().sum()
    total_patients_tests_all_time = pat_tests.groupby('id_patients')['date_test'].nunique().sum()

    print('Unique patients:', unique_patients_initial)
    print('Total assessments in period:', unique_patients_assessments)
    print('All time tests for these patients:', total_patients_tests_all_time)

    # Remove assessments with invalid BMI data
    pat_tests = exclude_invalid_bmi(pat_tests)
    unique_patients_valid_bmi = pat_tests['id_patients'].nunique()
    unique_assessments_valid_bmi = pat_tests.groupby('id_patients')['created_at_assessments'].nunique().sum()
    print('After dropping invalid BMI:')
    print('Unique patients:', unique_patients_valid_bmi, '(',
          (unique_patients_initial - unique_patients_valid_bmi),'removed)')
    print('Total assessments in period:', unique_assessments_valid_bmi)

    # Remove proxy loggers
    pat_tests = pat_tests[pat_tests['reported_by_another'] == 0]
    unique_patients_no_proxy = pat_tests['id_patients'].nunique()
    unique_assessments_no_proxy = pat_tests.groupby('id_patients')['created_at_assessments'].nunique().sum()
    print('After dropping proxy logs:')
    print('Unique patients:', unique_patients_no_proxy, '(',
          (unique_patients_valid_bmi - unique_patients_no_proxy),'removed)')
    print('Total assessments in period:', unique_assessments_no_proxy)

    # Remove under 18s (YOB < 2003) & invalid ages
    pat_tests = pat_tests[pat_tests['year_of_birth'] < 2003]
    pat_tests = pat_tests[pat_tests['year_of_birth'] > 1910]
    unique_patients_adult = pat_tests['id_patients'].nunique()
    unique_assessments_adult = pat_tests.groupby('id_patients')['created_at_assessments'].nunique().sum()
    print('After dropping u18s/invalid ages:')
    print('Unique patients:', unique_patients_adult, '(',
          (unique_patients_no_proxy - unique_patients_adult),'removed)')
    print('Total assessments in period:', unique_assessments_adult)

    # Convert dates in dataframe to datetime
    pat_tests = convert_df_dates(pat_tests)

    # New dataframe filtered for tests in timeperiod
    pat_tests_in_time_period = pat_tests[pat_tests['date_test'] >= start_date]
    total_tests_in_period = pat_tests_in_time_period.groupby('id_patients')['date_test'].nunique().sum()
    print('Total tests in period:', total_tests_in_period)

    # Tests in period before free testing ends
    pat_tests_free = pat_tests[pat_tests['date_test'] < datetime.date(2022, 4, 1)]
    pat_tests_while_free = pat_tests_free[pat_tests_free['date_test'] >= start_date]
    total_tests_while_free = pat_tests_while_free.groupby('id_patients')['date_test'].nunique().sum()
    print('Tests before 1st April:', total_tests_while_free)

    # Tests in period after free testing ends
    pat_tests_after_free = pat_tests[pat_tests['date_test'] >= datetime.date(2022, 4, 1)]
    total_tests_after_free = pat_tests_after_free.groupby('id_patients')['date_test'].nunique().sum()
    print('Tests after 1st April:', total_tests_after_free)

    # Convert assessment dates to datetime
    pat_tests = convert_df_date_series(pat_tests, series='created_at_assessments')

    # Assessments in period before free testing ends
    assessments_while_free = pat_tests[pat_tests['created_at_assessments'] < datetime.date(2022, 4, 1)]
    total_assessments_while_free = assessments_while_free.groupby('id_patients')['created_at_assessments'].nunique().sum()
    print('Assessments before 1st April:', total_assessments_while_free)

    # Assessments after free testing ends
    assessments_after_free = pat_tests[pat_tests['created_at_assessments'] >= datetime.date(2022, 4, 1)]
    total_assessments_after_free = assessments_after_free.groupby('id_patients')['created_at_assessments'].nunique().sum()
    print('Assessments from 1st April:', total_assessments_after_free)

    # Patients assessing before and after
    patients_assessing_while_free = assessments_while_free['id_patients'].nunique()
    print('Patients providing assessments while free:', patients_assessing_while_free)
    patients_assessing_after_free = assessments_after_free['id_patients'].nunique()
    print('Patients providing assessments after free:', patients_assessing_after_free)

    # Apply sliding window inclusion criteria
    start_date = datetime.date(2022,1,1)
    end_date = datetime.date(2022,5,23)
    window = 7
    ratio_of_weeks = 2/3

    # Get a dataframe with id_patients and acceptance boolean
    patients_included_excluded = return_dataframe_of_accepted_patients(pat_tests, start_date, end_date, window, ratio_of_weeks)
    print('patients included excluded type:', type(patients_included_excluded))
    print('new df keys:', patients_included_excluded.keys())

    # Merge with the dataframe
    print('pat_tests type', type(pat_tests))
    pat_tests.merge(patients_included_excluded, how='inner', on='id_patients')
    print('keys:', pat_tests.keys())

    # Number of patients included/excluded
    print(pat_tests.groupby(['meets_criteria'])['id_patients'].nunique())

    # Exclude patients who don't meet criteria
    pat_tests = pat_tests[pat_tests['meets_criteria'] == True]
    unique_patients_frequent = pat_tests['id_patients'].nunique()
    unique_assessments_frequent = pat_tests.groupby('id_patients')['created_at_assessments'].nunique().sum()
    print('After dropping infrequent contributors:')
    print('Unique patients:', unique_patients_frequent, '(',
      (unique_patients_adult - unique_patients_frequent), 'removed)')
    print('Total assessments in period:', unique_assessments_frequent)

    print('For included patients:')
    print_before_after_summary(pat_tests)

    # Look at proportions of healthcare workers
    # Look at proportion of asthmatics, high BMI, diabetics, chemotherapy patients


def correlation_matrix(df):
    # Print a spearman correlation matrix
    corr_matrix = df.corr(method = 'spearman')
    print(corr_matrix)
    print('done')
    return corr_matrix

if __name__ == '__main__':
    main('/nvme1_mounts/nvme1lv02/coneill/data.csv',
         start_date = datetime.date(2022, 2, 1))
