import datetime
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
import datetime as dt
import scipy.stats as stats
import csv

import my_functions as mf
from my_functions import exclude_invalid_bmi
from my_functions import convert_df_dates
from my_functions import convert_df_date_series
from my_functions import assessment_frequency_criteria
from my_functions import return_dataframe_of_accepted_patients
from my_functions import print_before_after_summary
from my_functions import print_summary
from my_functions import compare_freq_around_policy_change
from my_functions import compare_median_freq_around_policy_change
from my_functions import return_df_assessments_tests_daily
from my_functions import plot_daily_assess_tests
from my_functions import demographic_info
from my_functions import write_df_to_csv

print('imports complete')

def main(path, start_date: datetime):
    print('running')
    print('Period starts:', start_date)

    # Import the tests csv file into a dataframe
    pat_tests = pd.read_csv(path)

    # Create a smaller version for testing with
    # pat_tests = pat_tests.iloc[:50]

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

    #####
    # Sliding window assessment
    # Commented out until working
    #####

    # Apply sliding window inclusion criteria
    start_date = datetime.date(2022,1,1)
    end_date = datetime.date(2022,5,23)
    window = 7
    ratio_of_weeks = 2/3

    # Get a dataframe with id_patients and acceptance boolean
    patients_included_excluded = return_dataframe_of_accepted_patients(pat_tests, start_date, end_date, window, ratio_of_weeks)
    print('patients included excluded type:', type(patients_included_excluded))
    print('new df keys:', patients_included_excluded.keys())

    # Merge with the existing dataframe
    print('pat_tests type', type(pat_tests))
    pat_tests_with_inclusion = pat_tests.merge(patients_included_excluded, how='inner', on='id_patients')
    print('keys:', pat_tests_with_inclusion.keys())

    # Number of patients included/excluded
    print(pat_tests_with_inclusion.groupby(['meets_criteria'])['id_patients'].nunique().sum())

    # Exclude patients who don't meet criteria
    pat_tests = pat_tests_with_inclusion[pat_tests_with_inclusion['meets_criteria'] == True]
    unique_patients_frequent = pat_tests['id_patients'].nunique()
    unique_assessments_frequent = pat_tests.groupby('id_patients')['created_at_assessments'].nunique().sum()
    print('After dropping infrequent contributors:')
    print('Unique patients:', unique_patients_frequent, '(',
      (unique_patients_adult - unique_patients_frequent), 'removed)')
    print('Total assessments in period:', unique_assessments_frequent)

    print('For included patients:')
    print_before_after_summary(pat_tests)

    ####
    # Above: sliding window
    ####

    # Add in the countries data (mapping to lsoa11cd)
    country_mapping = pd.read_csv('/nvme1_mounts/nvme1lv02/coneill/project_v4/countries_csv.csv')
    # Edit the lsoa11cd string in original dataframe
    pat_tests['lsoa11cd_str'] = pat_tests['lsoa11cd'].apply(lambda x: x[2:-1])
    # pat_tests_with_country = pat_tests_with_inclusion.merge(country_mapping, how='inner', on='lsoa11cd')
    pat_tests_with_country = pat_tests.merge(country_mapping, how='inner', left_on='lsoa11cd_str', right_on='lsoa11cd')
    # Tidy up by removing surplus lsoa... columns
    pat_tests_with_country = pat_tests_with_country.drop(columns=['lsoa11cd_x', 'lsoa11cd_y'])
    pat_tests_with_country = pat_tests_with_country.rename(columns={"lsoa11cd_str": "lsoa11cd"})


    # Overall numbers in each country
    pat_tests_england = pat_tests_with_country[(pat_tests_with_country['country'] == 'England')]
    pat_tests_wales = pat_tests_with_country[pat_tests_with_country['country'] == 'Wales']
    pat_tests_scotland = pat_tests_with_country[pat_tests_with_country['country'] == 'Scotland']
    pat_tests_n_ireland = pat_tests_with_country[pat_tests_with_country['country'] == 'Northern Ireland']

    print_summary(pat_tests_england, df_name='England')
    print_summary(pat_tests_wales, df_name='Wales')
    print_summary(pat_tests_scotland, df_name='Scotland')
    print_summary(pat_tests_n_ireland, df_name='Northern Ireland')

    # Explore England before and after policy change
    print('England only data summary:')
    print_before_after_summary(pat_tests_england)

    # Compare frequency histograms before vs after
    
    england_patients_mean = compare_freq_around_policy_change(dataframe=pat_tests_england,
                                      start_date = datetime.date(2022,2,1),
                                      end_date = datetime.date(2022,5,23),
                                      policy_change_date = datetime.date(2022,4,1))

    england_patients_median = compare_median_freq_around_policy_change(dataframe=pat_tests_england,
                                                              start_date=datetime.date(2022, 2, 1),
                                                              end_date=datetime.date(2022, 5, 23),
                                                              policy_change_date=datetime.date(2022, 4, 1))

    # Fill NaNs with 0... as NaNs appear when patients have not done a single test
    england_patients_mean = england_patients_mean.fillna(0)
    england_patients_median = england_patients_median.fillna(0)

    print(england_patients_mean)
    print(england_patients_median)

    # Plot a histogram comparing frequency distributions
    england_patients_mean.plot.hist(bins=10, alpha=0.5)
    england_patients_median.plot.hist(bins=10, alpha=0.5)

    plt.show()

    stat, pvalue = stats.wilcoxon(england_patients_mean['mean_before'], england_patients_mean['mean_after'])
    # could try feeding with .values ... effectively feeds as an array

    print('Mean averages:')
    print('Statistic:', stat)
    print('P-value:', pvalue)

    stat_med, pvalue_med = stats.wilcoxon(england_patients_median['median_before'], england_patients_median['median_after'])

    print('Median averages:')
    print('Statistic:', stat_med)
    print('P-value:', pvalue_med)

    # Assess overall assessments and tests for the period (in England)
    assess_tests_england_daily = return_df_assessments_tests_daily(pat_tests_england)
    fig = plot_daily_assess_tests(assess_tests_england_daily, country="England")
    fig.show()

    # Assess overall assessments and tests for the period (in Scotland)
    assess_tests_scotland_daily = return_df_assessments_tests_daily(pat_tests_scotland)
    fig = plot_daily_assess_tests(assess_tests_scotland_daily, country="Scotland")
    fig.show()

    # Merge daily ratios for all the countries
    assess_tests_wales_daily = return_df_assessments_tests_daily(pat_tests_wales)
    assess_tests_ni_daily = return_df_assessments_tests_daily(pat_tests_n_ireland)
    tests_assessments_ratios = assess_tests_england_daily.merge(assess_tests_scotland_daily,
                                                               how='inner', on='date', suffixes=('_eng', '_sco'))
    tests_assessments_ratios = tests_assessments_ratios.merge(assess_tests_wales_daily,
                                                              how='inner', on='date')
    tests_assessments_ratios = tests_assessments_ratios.filter(columns = ['date', 'ratio_eng', 'ratio_sco', 'ratio'])
    tests_assessments_ratios = tests_assessments_ratios.rename({'ratio':'ratio_wal'})
    tests_assessments_ratios = tests_assessments_ratios.merge(assess_tests_ni_daily,
                                                              how='inner', on='date')
    tests_assessments_ratios = tests_assessments_ratios.filter(columns=['date', 'ratio_eng', 'ratio_sco',
                                                                        'ratio_wal', 'ratio'])
    tests_assessments_ratios = tests_assessments_ratios.rename({'ratio': 'ratio_ni'})




    # Look at proportions of healthcare workers
    # Look at proportion of asthmatics, high BMI, diabetics, chemotherapy patients
    # Normalising the rate of change per individual

    counts_df, chemo_df, contact_df, asthmatics_df, bmi_df, gender_df = demographic_info(pat_tests_england)
    print(counts_df)
    print(chemo_df)
    print(asthmatics_df)
    print(bmi_df)
    print(gender_df)

    # Write the dataframe of patients/assessments/tests meeting acceptance criteria to file
    write_df_to_csv(pat_tests_england, 'pat_tests_england_a.csv')
    write_df_to_csv(pat_tests_scotland, 'pat_tests_scotland_a.csv')
    write_df_to_csv(pat_tests_wales, 'pat_tests_wales_a.csv')
    write_df_to_csv(pat_tests_n_ireland, 'pat_tests_ni_a.csv')

    # pat_tests_acceptance = pat_tests_england
    # columns = pat_tests_acceptance.columns
    # values = pat_tests_acceptance.values
    # filename = 'pat_tests_england.csv'
    #
    # with open(filename, 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(columns)
    #     writer.writerows(values)



    print('file saved')

if __name__ == '__main__':
    main(#'/nvme1_mounts/nvme1lv02/coneill/data.csv',
         #'/nvme1_mounts/nvme1lv02/coneill/project_v4/pat_tests_truncated',
         '/nvme1_mounts/nvme1lv02/coneill/project_v4/data_05_30_a.csv',
         start_date = datetime.date(2022, 2, 1))
