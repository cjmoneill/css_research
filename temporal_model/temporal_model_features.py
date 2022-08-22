import datetime
import pandas as pd

from my_functions import return_df_of_totals_relative, convert_df_dates, periods_symptomatic, return_df_of_assessments
from my_functions import write_df_to_csv, convert_df_date_series, symptoms_in_period, skeleton_features_df


# Take the tests table from each country as an input

# Create a function to calculate total tests in a period (start date, end date) and add to new dataframe
# with rows for patient ID, period relative to change, tests in period (i.e. multiple rows per patient)
# Repeat for all patients to concatenate a new dataframe

def main(path, start_date, end_date, policy_change_date):

    # Read in the dataframe
    dataframe = pd.read_csv(path)
    dataframe = convert_df_dates(dataframe)
    dataframe = convert_df_date_series(dataframe, series='created_at_assessments')

    # Get a skeleton dataframe covering all patients/periods
    skeleton_df = skeleton_features_df(dataframe, start_date, end_date, policy_change_date)

    # Get new dataframe with tests per period per patient
    patient_tests_per_period = return_df_of_totals_relative(dataframe, start_date, end_date, policy_change_date)
    print(patient_tests_per_period.head(50))

    # Get a new dataframe showing whether patients were symptomatic & days symptomatic per period per patient
    patient_symptomatic = periods_symptomatic(dataframe, start_date, end_date, policy_change_date)
    print(patient_symptomatic.head(50))

    # Get a new dataframe showing assessments per period
    patient_assessments = return_df_of_assessments(dataframe, start_date, end_date, policy_change_date)
    print(patient_assessments.head(50))

    # Get a new dataframe showing which symptoms patients had, and sum of unique symptoms per period

    symptoms_detail = symptoms_in_period(dataframe, start_date, end_date, policy_change_date)
    print(symptoms_detail.head(50))

    # Merge dataframes

    assess = pd.merge(skeleton_df, patient_assessments, how='outer', left_on=['id_patients', 'period_relative_to_change'],
                      right_on=['id_patients', 'period_relative_to_change'])

    assess_test = pd.merge(assess, patient_tests_per_period, how='outer', left_on=['id_patients', 'period_relative_to_change'],
                      right_on=['id_patients', 'period_relative_to_change'])

    assess_test_sympt = pd.merge(assess_test, patient_symptomatic, how='outer', left_on=['id_patients', 'period_relative_to_change'],
                      right_on=['id_patients', 'period_relative_to_change'])

    new_df = pd.merge(assess_test_sympt, symptoms_detail, how='outer', left_on=['id_patients', 'period_relative_to_change'],
                      right_on=['id_patients', 'period_relative_to_change'])

    print(new_df.head(100))

    new_filename = 'temporal_features_detail_test'
    write_df_to_csv(new_df, new_filename)
    print('complete')


if __name__ == '__main__':
    main('/nvme1_mounts/nvme1lv02/coneill/project_v4/merged_pat_tests_england.csv',
         start_date = datetime.date(2022, 2, 1),
         end_date = datetime.date(2022,5,30),
         policy_change_date = datetime.date(2022,4,1))
