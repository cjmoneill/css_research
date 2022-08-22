import scipy.stats
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import wilcoxon
from my_functions import print_summary, print_before_after_summary, convert_df_dates, convert_df_date_series, stratify_ages, add_bmi_catgory_column
from my_functions import compare_freq_around_policy_change, compare_tests_around_policy_change, compare_median_freq_around_policy_change
from my_functions import plot_daily_assess_tests, return_df_assessments_tests_daily, demographic_info, age_distribution, plot_pos_test_ratio
from scipy import stats

def main(path):

    # Read in dataframe & convert dates
    print("Reading in data")
    pat_tests = pd.read_csv(path)
    print("Read in complete")

    print(pat_tests.keys())

    # Convert dates in dataframe to datetime
    pat_tests = convert_df_dates(pat_tests)
    pat_tests = convert_df_date_series(pat_tests, series='created_at_assessments')

    # Assess ratio of positive tests
    unique_tests = pat_tests.drop_duplicates(subset = ['id_patients', 'date_test'], keep='last')
    positives = unique_tests[unique_tests['result'] == 4]
    others = unique_tests[unique_tests['result'] != 4]
    pos_by_date = positives['date_test'].value_counts()
    oth_by_date = others['date_test'].value_counts()
    print(pos_by_date)
    print(oth_by_date)
    pos = pd.DataFrame(pos_by_date)
    pos = pos.reset_index()
    pos = pos.rename(columns={"date": "total_pos"})
    print(pos.keys())
    oth = pd.DataFrame(oth_by_date)
    oth = oth.reset_index()
    oth = oth.rename(columns={"date": "total_oth"})
    print(oth.keys())

    comb = pd.merge(pos, oth, how='outer', on='index', suffixes=('_pos', '_oth'))
    comb['ratio'] = comb['date_test_pos'] / (comb['date_test_pos'] + comb['date_test_oth'])
    comb_for_period = comb[comb['index'] >= datetime.date(2022,2,1)]

    print(len(comb_for_period))
    print(comb_for_period.head(20))

    comb_for_period = convert_df_date_series(comb_for_period, series='index')
    comb_for_period = comb_for_period.rename(columns={"index": "day"})
    print(comb_for_period.head(20))
    comb_for_period = comb_for_period.sort_values(by='day')
    print(comb_for_period.head(20))
    fig = plot_pos_test_ratio(comb_for_period)
    fig.show()


    # unique_tests = pat_tests.drop_duplicates(subset=['id_patients', 'date_test', 'mechanism'], keep='last')
    unique_tests = pat_tests
    # unique_tests = unique_tests[unique_tests['date_test'] >= datetime.date(2022,2,1)]
    # unique_tests_after = unique_tests[unique_tests['date_test'] >= datetime.date(2022,4,1)]

    # Add age category
    pat_tests = stratify_ages(pat_tests, key='age_2022_start')

    lower_higher = compare_tests_around_policy_change(pat_tests,
                                                      start_date = datetime.date(2022,2,1),
                                                      end_date = datetime.date(2022,5,23),
                                                      policy_change_date= datetime.date(2022,4,1))

    print('lower higher totals')
    print(lower_higher['lower_stable_higher'].value_counts())

    lower = lower_higher[lower_higher['lower_stable_higher'] == 0]
    print_before_after_summary(lower)


    # Explore gender
    print('Test from females only:')
    female = unique_tests[unique_tests['gender'] == 1]
    print_before_after_summary(female)

    # Explore contact health
    print('Test from contact health workers only:')
    contact = unique_tests[unique_tests['contact_health_worker'] == 1]
    print_before_after_summary(contact)

    # Explore preconditions
    preconditions = pat_tests[
        ['id_patients', 'has_cancer', 'has_asthma', 'does_chemotherapy', 'has_heart_disease', 'has_lung_disease']]

    preconditions = preconditions.drop_duplicates()

    conditions = [preconditions.has_cancer == 2,
                  preconditions.has_asthma == 2,
                  preconditions.does_chemotherapy == 2,
                  preconditions.has_heart_disease == 2,
                  preconditions.has_lung_disease == 2]

    value = [1, 1, 1, 1, 1]

    dest = pat_tests

    preconditions['has_preconditions'] = np.select(conditions, value)
    preconditions_sum = pd.merge(dest, preconditions[['id_patients', 'has_preconditions']], on='id_patients', how='left')

    print(preconditions_sum['has_preconditions'].value_counts())
    print(preconditions_sum.keys())
    print(preconditions_sum.head(100))

    # preconditions_sum = preconditions_sum.groupby('id_patients')['has_preconditions'].unique()
    # preconditions_sum = preconditions_sum.reset_index()
    # preconditions_sum['preconditions_status'] = preconditions_sum['has_preconditions'].apply(
    #     lambda x: 1 if max(x) > 0 else 0)
    # preconditions_sum = preconditions_sum.drop(columns='has_preconditions')
    #
    # df = df.merge(preconditions_sum, how="inner", on='id_patients')
    # print(df)
    print('Preconditions only')
    preconditions_only = preconditions_sum[preconditions_sum['has_preconditions'] == 1]
    print_before_after_summary(preconditions_only)

    print('Rapid tests only')
    rapid_tests = unique_tests[unique_tests['is_rapid_test'] == 2]
    print_before_after_summary(rapid_tests)

    print('Positive test results only')
    pos_test_results = unique_tests[unique_tests['result'] == 4]
    print_before_after_summary(pos_test_results)


    # Explore age categories
    print('Ages before and after')
    print('<25')
    age_group_0 = unique_tests[unique_tests['age_category'] == 0]
    print_before_after_summary(age_group_0)

    print('25-39')
    age_group_1 = unique_tests[unique_tests['age_category'] == 1]
    print_before_after_summary(age_group_1)

    print('40-54')
    age_group_2 = unique_tests[unique_tests['age_category'] == 2]
    print_before_after_summary(age_group_2)

    print('55-69')
    age_group_3 = unique_tests[unique_tests['age_category'] == 3]
    print_before_after_summary(age_group_3)

    print('Over 70')
    age_group_4 = unique_tests[unique_tests['age_category'] == 4]
    print_before_after_summary(age_group_4)

    # Explore BMI categories

    unique_tests = add_bmi_catgory_column(unique_tests)
    print('Underweight')
    bmi_group_0 = unique_tests[unique_tests['bmi_category'] == 0]
    print_before_after_summary(bmi_group_0)

    print('Healthy')
    bmi_group_1 = unique_tests[unique_tests['bmi_category'] == 1]
    print_before_after_summary(bmi_group_1)

    print('Overweight')
    bmi_group_2 = unique_tests[unique_tests['bmi_category'] == 2]
    print_before_after_summary(bmi_group_2)

    print('Obese')
    bmi_group_3 = unique_tests[unique_tests['bmi_category'] == 3]
    print_before_after_summary(bmi_group_3)


    print(unique_tests.keys())
    print('Unique test:', len(unique_tests))
    # print('Unique tests after:', len(unique_tests_after))
    print(unique_tests['mechanism'].value_counts())
    print(unique_tests['is_rapid_test'].value_counts())
    print(unique_tests['test_doy'].value_counts())

    print_summary(pat_tests, df_name='England')
    print_before_after_summary(pat_tests)

    counts_df, chemo_df, contact_df, asthmatics_df, bmi_df, gender_df, cancer_df, hd_df, ld_df = demographic_info(pat_tests)
    print(counts_df)
    print(chemo_df)
    print(asthmatics_df)
    print(bmi_df)
    print(gender_df)
    print(cancer_df)
    print(hd_df)
    print(ld_df)
    print(contact_df)

    # Show distribution of ages
    age_dist = age_distribution(pat_tests)
    print(age_dist)

    # Get contact health worker info

    # Compare frequency histograms before vs after

    england_patients_mean = compare_freq_around_policy_change(dataframe=pat_tests,
                                                              start_date=datetime.date(2022, 2, 1),
                                                              end_date=datetime.date(2022, 5, 23),
                                                              policy_change_date=datetime.date(2022, 4, 1))

    england_patients_median = compare_median_freq_around_policy_change(dataframe=pat_tests,
                                                                       start_date=datetime.date(2022, 2, 1),
                                                                       end_date=datetime.date(2022, 5, 23),
                                                                       policy_change_date=datetime.date(2022, 4, 1))

    # Fill NaNs with 0... as NaNs appear when patients have not done a single test
    england_patients_mean = england_patients_mean.fillna(0)
    england_patients_median = england_patients_median.fillna(0)

    print(england_patients_mean)
    print(england_patients_median)

    print('desc:', england_patients_mean[['mean_before', 'mean_after']].describe())
    print('desc:', england_patients_median[['median_before', 'median_after']].describe())

    # Plot a histogram comparing frequency distributions
    print("Mean before after")
    mean_before = england_patients_mean['mean_before']
    mean_after = england_patients_mean['mean_after']

    mean_before_totals = mean_before.value_counts()
    print(mean_before_totals)
    mean_after_totals = mean_after.value_counts()
    print(mean_after_totals)

    # plt.bar(mean_before_totals)
    # plt.title('Mean weekly tests before change')
    # plt.axis([0, 7, 0, 16000])
    # plt.show()
    # plt.bar(mean_after_totals)
    # plt.title('Mean weekly tests after change')
    # plt.axis([0, 7, 0, 16000])
    # plt.show()

    print("Medians before after")
    median_before = england_patients_median['median_before']
    median_after = england_patients_median['median_after']

    median_before_totals = median_before.value_counts()
    print(median_before_totals)
    median_after_totals = median_after.value_counts()
    print(median_after_totals)

    # plt.bar(median_before_totals)
    # plt.title('Median weekly tests before change')
    # plt.axis([0, 7, 0, 16000])
    # plt.show()
    # plt.bar(median_after_totals)
    # plt.title('Median weekly tests after change')
    # plt.axis([0, 7, 0, 16000])
    # plt.show()

    # england_patients_mean.plot.hist(bins=[0,1,2,3,4,5,7], alpha=0.5)
    # plt.axis([0, 7, 0, 20000])
    # plt.show()

    x = england_patients_mean['mean_before']
    y = england_patients_mean['mean_after']
    bins = [0,1,2,3,4,5,6,7]
    plt.hist([x, y], bins, label=['Mean before', 'Mean after'])
    plt.legend(loc='upper right')
    plt.axis([0, 7, 0, 20000])
    plt.show()

    # england_patients_median.plot.hist(bins=[0,1,2,3,4,5,6,7], alpha=0.5)
    # plt.axis([0, 7, 0, 20000])
    # plt.show()

    x = england_patients_median['median_before']
    y = england_patients_median['median_after']
    bins = [0,1,2,3,4,5,6,7]
    plt.hist([x, y], bins, label=['Before policy change', 'After policy change'])
    plt.title('Median number of weekly tests per patient', fontsize=14)
    plt.ylabel('Total participants', fontsize=12)
    plt.xlabel('Median tests per week', fontsize=12)
    plt.legend(loc='upper right')
    plt.axis([0, 7, 0, 20000])
    plt.show()

    # Wilcoxon on means... requires 1D arrays
    mean_before = england_patients_mean['mean_before'].to_numpy()
    mean_after = england_patients_mean['mean_after'].to_numpy()
    stat, pvalue = scipy.stats.wilcoxon(mean_before, mean_after)
    # could try feeding with .values ... effectively feeds as an array
    print('Mean averages:')
    print('Statistic:', "{:.20f}".format(stat))
    print('P-value:', "{:.20f}".format(pvalue))
    # print('Z:', z)

    print(stats.wilcoxon(england_patients_mean['mean_before'], england_patients_mean['mean_after']))

    # Wilcoxon on medians
    median_before = england_patients_median['median_before'].to_numpy()
    median_after = england_patients_median['median_after'].to_numpy()
    stat_med, pvalue_med = scipy.stats.wilcoxon(median_before, median_after)

    print(stats.wilcoxon(england_patients_median['median_before'], england_patients_median['median_after']))


    print('Median averages:')
    print('Statistic:', "{:.20f}".format(stat_med))
    print('P-value:', "{:.20f}".format(pvalue_med))
    # print('Z:', z)

    # Assess overall assessments and tests for the period (in England)
    assess_tests_england_daily = return_df_assessments_tests_daily(pat_tests)
    fig = plot_daily_assess_tests(assess_tests_england_daily, country="England")
    fig.show()




if __name__ == '__main__':
    main(path = '/nvme1_mounts/nvme1lv02/coneill/project_v4/merged_pat_tests_england.csv')