import scipy.stats
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from my_functions import convert_df_dates, convert_df_date_series, compare_median_freq_around_policy_change, stratify_ages
from my_functions import  write_df_to_csv, compare_freq_around_policy_change
def main(path):

    # Read in dataframe & convert dates
    print("Reading in data")
    pat_tests = pd.read_csv(path)
    print("Read in complete")
    print("Keys:", pat_tests.keys())

    ### Get medians for further analysis
    # # Convert dates in dataframe to datetime
    # pat_tests = convert_df_dates(pat_tests)
    # pat_tests = convert_df_date_series(pat_tests, series='created_at_assessments')
    #
    # # Get patient ages and add age group
    # pat_ages = pat_tests.filter(items=['id_patients', 'age_2022_start'])
    # pat_ages = stratify_ages(pat_ages, key='age_2022_start')
    #
    # # Get median test values
    # medians_pp = compare_median_freq_around_policy_change(dataframe=pat_tests,
    #                                                                    start_date=datetime.date(2022, 2, 1),
    #                                                                    end_date=datetime.date(2022, 5, 23))                                                                   policy_change_date=datetime.date(2022, 4, 1))
    # Add age group
    # pat_ages_medians = pd.merge(pat_ages, medians_pp, how='left', on='id_patients')
    # filename = 'further_analysis.csv'
    # write_df_to_csv(pat_ages_medians, filename)

    ## Get means for further analysis
    # Convert dates in dataframe to datetime
    # pat_tests = convert_df_dates(pat_tests)
    # pat_tests = convert_df_date_series(pat_tests, series='created_at_assessments')
    #
    # # Get patient ages and add age group
    # pat_ages = pat_tests.filter(items=['id_patients', 'age_2022_start'])
    # pat_ages = stratify_ages(pat_ages, key='age_2022_start')
    #
    # # Get median test values
    # means_pp = compare_freq_around_policy_change(dataframe=pat_tests,
    #                                                start_date=datetime.date(2022, 2, 1),
    #                                                end_date=datetime.date(2022, 5, 23),
    #                                                policy_change_date=datetime.date(2022, 4, 1))
    #
    # # Add age group
    # pat_ages_means = pd.merge(pat_ages, means_pp, how='left', on='id_patients')
    # filename = 'further_analysis_means.csv'
    # write_df_to_csv(pat_ages_means, filename)


    print(len(pat_tests))
    uniques = pat_tests.drop_duplicates(keep='last')

    print(len(uniques))

    group_0 = uniques[uniques['age_category'] == 0]
    group_1 = uniques[uniques['age_category'] == 1]
    group_2 = uniques[uniques['age_category'] == 2]
    group_3 = uniques[uniques['age_category'] == 3]
    group_4 = uniques[uniques['age_category'] == 4]

    # def print_hist(group, label='group'):
    #     x = group['median_before']
    #     y = group['median_after']
    #     bins = [0, 1, 2, 3, 4, 5, 6, 7]
    #     plt.hist([x, y], bins, label=['Median before', 'Median after'])
    #     plt.legend(loc='upper right')
    #     plt.title('Median tests per week group {}'.format(label), fontsize=14)
    #     plt.ylabel('Total participants')
    #     plt.xlabel('Median tests per week')
    #     plt.axis([0, 7, 0, 7500])
    #     plt.show()

    # Plot histograms of weekly distributions
    # print_hist(group_0, "0")
    # print_hist(group_1, "1")
    # print_hist(group_2, "2")
    # print_hist(group_3, "3")
    # print_hist(group_4, "4")

    # print(len(group_0))
    # print('0',group_0[['median_before', 'median_after']].mean())
    # print('1',group_1[['median_before', 'median_after']].mean())
    # print('2',group_2[['median_before', 'median_after']].mean())
    # print('3',group_3[['median_before', 'median_after']].mean())
    # print('4',group_4[['median_before', 'median_after']].mean())

    print(len(group_0))
    print('0',group_0[['mean_before', 'mean_after']].mean())
    print('1',group_1[['mean_before', 'mean_after']].mean())
    print('2',group_2[['mean_before', 'mean_after']].mean())
    print('3',group_3[['mean_before', 'mean_after']].mean())
    print('4',group_4[['mean_before', 'mean_after']].mean())


    # Group by ages
    # grouped = pat_ages_medians.groupby('age_category')



if __name__ == '__main__':
    main(path = '/nvme1_mounts/nvme1lv02/coneill/project_v4/further_analysis_means.csv')