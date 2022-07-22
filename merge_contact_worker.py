# To merge contact worker status with existing patient dataframes
import pandas as pd
from my_functions import write_df_to_csv

def bit_encode_str(x):
    x = "b'{}'".format(x)
    return x

def main(patients_path, dataframe_path):

    patient_info = pd.read_csv(patients_path)
    working_dataframe = pd.read_csv(dataframe_path)
    pat_head = patient_info.head(20)
    work_head = working_dataframe.head(20)
    print(pat_head)
    print(work_head)

    # Get healthcare worker status
    print(patient_info.keys())
    print(working_dataframe.keys())

    patient_info = patient_info.filter(['id', 'contact_health_worker'], axis='columns')
    patient_info['id_patients'] = patient_info['id']#.apply(lambda x: bit_encode_str(x))
    working_dataframe['id_patients'] = working_dataframe['id_patients'].apply(lambda x: x.split("'")[1])
    print(patient_info['id_patients'].value_counts())
    print(patient_info['contact_health_worker'].value_counts())
    df = working_dataframe.merge(patient_info, how='inner', left_on='id_patients', right_on='id_patients')
    # df = df.drop(columns='id_y')
    print(df.keys())

    new_filename = dataframe_path.split('/')
    new_filename = new_filename[-1]
    new_filename = 'merged_{}'.format(new_filename)
    print(new_filename)

    write_df_to_csv(df, new_filename)
    print('complete')

if __name__ == '__main__':
    main('/hdd_mounts/archivelv/input_data/patients_export_geocodes_20220425040028.csv.gz',
         '/nvme1_mounts/nvme1lv02/coneill/project_v4/pat_tests_england.csv')