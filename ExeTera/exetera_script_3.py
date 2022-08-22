import exetera
import pandas as pd
from exetera.core.session import Session
from numba import njit
import exetera.core.dataframe as df
import exetera.core.fields as fld
import exetera.core.operations as ops
import numpy as np
import csv
from datetime import datetime

# List of symptoms that we want to include
list_symptoms = ['location', 'treatment', 'altered_smell', 'brain_fog',
                 'chest_pain', 'chills_or_shivers', 'diarrhoea',
                  'dizzy_light_headed', 'fatigue', 'fever', 'headache',
                  'loss_of_smell', 'nausea', 'persistent_cough',
                  'runny_nose', 'shortness_of_breath', 'sneezing', 'sore_throat',
                  'patient_id', 'created_at']

# Patient fields relevant to the analysis
patient_fields = ['id', 'year_of_birth', 'country_code', 'bmi', 'ethnicity', 'gender',
                  'lsoa11cd', 'reported_by_another', 'has_asthma', 'has_hayfever',
                  'has_cancer', 'has_diabetes', 'has_heart_disease', 'has_kidney_disease', 'has_lung_disease',
                  'does_chemotherapy', 'takes_immunosuppressants', 'contact_health_worker']

test_fields = ['patient_id', 'created_at', 'date_taken_specific', 'mechanism', 'result', 'is_rapid_test']

# Function to convert date information
def get_ts_str(d):
    if d > 0:
        return datetime.fromtimestamp(d).strftime("%Y-%m-%d")
    else:
        return '0'

# Function to save a dataframe to a csv file
def save_df_to_csv(df, csv_name, fields, chunk=200000):  # chunk=100k ~ 20M/s
    with open(csv_name, 'w', newline='') as csvfile:
        columns = list(fields)
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        field1 = columns[0]
        for current_row in range(0, len(df[field1].data), chunk):
            to_row = current_row + chunk if current_row + chunk < len(df[field1].data) else len(df[field1].data)
            batch = list()
            for k in fields:
                if isinstance(df[k], fld.TimestampField):
                    batch.append([get_ts_str(d) for d in df[k].data[current_row:to_row]])
                else:
                    batch.append(df[k].data[current_row:to_row])
            writer.writerows(list(zip(*batch)))


# Function to take the 'positive' symptoms, in the case of multiple assessments where
# the presence of a symptom changes
@njit
def reduce_symptom(onesymp, sum, spans):
    for i in range(len(spans) - 1):
        onesymp[spans[i]] = 1 if np.any(onesymp[spans[i]:spans[i + 1]] > 1) else 0
        sum[i] |= onesymp[spans[i]]
    return onesymp, sum


def get_assess_test_vacc(src_filename: str,
                         dst_filename: str,
                         start_date: datetime,
                         included_patients_path: str,
                         save_name: str = 'data_tests_symptoms'):


    # Start an exetera session
    with Session() as s:
        # Open the dataset specified in read, specify where my data will be written
        src = s.open_dataset(src_filename, 'r', 'src')
        dst = s.open_dataset(dst_filename, 'w', 'dst')

        print('Setting up patients dataframe')
        # Create new dataframe in dst
        patients_df = dst.create_dataframe('patients_df')
        # Add fixed string field
        patients_field = patients_df.create_fixed_string('id_patients', length=100)
        # Read in included patients to pandas dataframe
        included_patients = pd.read_csv(included_patients_path)
        included_patients_series = included_patients['id_patients']
        # Copy data into new field
        patients_field.data[:].write([s.encode() for s in included_patients_series])

        # field = Session.create_fixed_string(group=dst, name='included_patients', length=100)
        # dst.create_dataframe('included_patients')
        # exetera.io.parsers.read_csv(csv_file=included_patients_path, ddf=included_patients, schema_dictionary=None,
        #                             schema_file=None)

        dst_included_patients = dst['patients_df']

        print(dst_included_patients)
        print(dst_included_patients.keys())

        # Create assessments dataframe from source
        print('Getting assessments and filtering')
        src_assessments = src['assessments']

        # Create a new dataframe for the assessments
        assessments = dst.create_dataframe('assessments')

        # For items in the list of symptoms, copy the data into the new dataframe
        for f in list_symptoms:
            df.copy(src_assessments[f], assessments, f)

        # Filter for the timeperiod
        # Specify date and convert to a timestamp
        dt = start_date
        seconds = dt.timestamp()
        # Create filter
        filter_date = (assessments['created_at'].data[:] >= seconds)
        # Apply filter
        assessments.apply_filter(filter_date)

        # Rename the location series
        df.move(assessments['location'], assessments, 'hospitalisation')

        print("Getting patients and filtering")

        # Define a dataframe from original data (just work with patients' data)
        src_patients = src['patients']

        # Create a new destination dataframe
        patients = dst.create_dataframe('patients')

        # Copy the assessments and patient IDs into the new dataframe
        for f in patient_fields:
            df.copy(src_patients[f], patients, f)

        # Create destination for merged patients (acceptance criteria only)
        accepted_patients = dst.create_dataframe('accepted_patients')

        # Merge with the included patients... (inner merge)
        df.merge(dst_included_patients, patients, dest=accepted_patients, how='left', left_on='id_patients', right_on='id',
                 left_suffix='_patients', right_suffix='_patients')

        # Create and apply a filter on country code (limit to GB).
        # Note that this is stored as 'b' not string.
        filter = accepted_patients['country_code'].data[:] == b'GB'
        accepted_patients.apply_filter(filter)

        # Add in the patient comorbidities
        # Create an array
        nr_comorbidity = np.zeros(len(accepted_patients['has_diabetes'].data))

        # Loop over the comorbidities, if value is 2 in the data, place 1 in new array
        # or else add a 0 (check the schema)
        for f in ['has_diabetes', 'has_heart_disease', 'has_lung_disease', 'does_chemotherapy', 'has_kidney_disease',
                  'has_cancer', 'takes_immunosuppressants', 'contact_health_worker']:
            nr_comorbidity += np.where(accepted_patients[f].data[:] == 2, 1, 0)

        # Add and copy over the comorbidities
        patients.create_numeric('nr_comorbidity', 'int32')
        patients['nr_comorbidity'].data.write(nr_comorbidity)
        df.move(patients['id'], patients, 'id_patients')


        # Get the test data
        print('Getting tests and filtering')

        # Create a new dataframe
        tests = dst.create_dataframe('sympt_tests')

        src_tests = src['tests']


        # Copy the assessments and patient IDs into the new dataframe
        for f in test_fields:
            df.copy(src_tests[f], tests, f)

        # Create and apply date filter
        filter_test = tests['created_at'].data[:] >= seconds
        tests.apply_filter(filter_test)


        print("Merging the assessments, tests and patients dataframes")
        # Merging the assessments to patients

        patients_tests = dst.create_dataframe('patients_assessments')

        df.merge(accepted_patients, tests, dest=patients_tests, how='inner', left_on='id_patients', right_on='patient_id',
                                        left_suffix='_patients', right_suffix='_tests')

        # print("Saving tests dataframe...")
        # save_df_to_csv(patients_tests, 'data_vacc_05_30_new.csv', list(patients_tests.keys()))

        patients_assessments_tests = dst.create_dataframe('patients_assessments_tests')

        df.merge(patients_tests, assessments, dest=patients_assessments_tests, how='left',left_on='id_patients', right_on='patient_id',
                                        left_suffix='_a', right_suffix='_assessments')


        # Getting out minimal information for the sliding window
        # patients_assessments_slid = dst.create_dataframe('patients_assessments_slid')
        # patients_assessments_columns = ['id_patients', 'created_at_assessments']
        # for f in patients_assessments_columns:
        #     df.copy(patients_assessments_tests[f], patients_assessments_slid, f)

        print('Saving full dataframe...')
        # Save dataframe to csv
        save_df_to_csv(patients_assessments_tests, 'included_05_30_new.csv', list(patients_assessments_tests.keys()))

        print('Complete!')


if __name__ == "__main__":
    srcfile = '/nvme0_mounts/nvme0lv01/exetera/recent/ds_20220530_full.hdf5'
    dstfile = 'data_v6.hdf5'
    start_date = datetime(2022, 2, 1)
    included_patients_path = '/nvme1_mounts/nvme1lv02/coneill/project_v4/included_patients.csv'
    get_assess_test_vacc(srcfile, dstfile, start_date, included_patients_path, 'tests_vac_symptoms.csv')
