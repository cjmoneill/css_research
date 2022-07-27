import datetime
import pandas as pd
import sys
sys.setrecursionlimit(3000)

from my_functions import convert_df_dates
from my_functions import compare_tests_around_policy_change
from my_functions import features_dataframe
from my_functions import stratified_sample_requirements
from my_functions import stratify_ages
from my_functions import create_stratified_test_sample

def main(path, start_date, end_date, policy_change_date):

    # Read in and create dataframe
    dataframe = pd.read_csv(path)
    dataframe = convert_df_dates(dataframe)

    # Create dataframe for target values
    dataframe_targets = compare_tests_around_policy_change(dataframe, start_date, end_date, policy_change_date)

    # Combine higher & stable into the same class
    dataframe_targets['lower_higher'] = dataframe_targets.lower_stable_higher.replace({2: 1})

    # Select the columns for the target variable
    dataframe_targets = dataframe_targets.filter(items=['id_patients', 'lower_higher'])

    # Create dataframe for features
    dataframe_features = features_dataframe(dataframe)

    # Merge to ensure the order is the same
    dataframe_merged = dataframe_features.merge(dataframe_targets, how='inner', on='id_patients')
    print(dataframe_merged)
    print('Unique patients:', dataframe_merged['id_patients'].nunique())
    print('Dataframe length', len(dataframe_merged))

   # Split into test and train datasets, stratified according to the features and classification

    # Add a column with age categories
    dataframe_merged = stratify_ages(dataframe_merged)

    # First get the ideal totals for each class in a test population
    stratified_sample_requirements(dataframe_merged, train_percentage=0.8, test_percentage=0.2)

    # Create dictionaries containing the ideal number of patients for each class to be sampled in the test set
    required_ages = {0: 10, 1: 197, 2: 743, 3: 1970, 4: 907}
    required_genders = {0: 2378, 1:1451}
    required_hw = {0: 1557, 1: 33}
    required_precs = {0: 3099, 1: 730}
    required_bmi = {0: 51, 1: 1729, 2: 1323, 3: 725}
    required_targets = {0: 2885, 1: 944}

    create_stratified_test_sample(dataframe_merged, required_ages, required_genders, required_hw,
                                                required_precs, required_bmi, required_targets)





if __name__ == '__main__':
    main('/nvme1_mounts/nvme1lv02/coneill/project_v4/merged_pat_tests_england.csv',

         start_date = datetime.date(2022, 2, 1),
         end_date = datetime.date(2022,5,30),
         policy_change_date = datetime.date(2022,4,1))