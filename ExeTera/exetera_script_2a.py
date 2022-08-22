from exetera.core.session import Session
from numba import njit
import exetera.core.dataframe as df
import exetera.core.fields as fld
import exetera.core.operations as ops
import numpy as np
import csv
from datetime import datetime

# List of symptoms that we want to include
list_symptoms = ['location', 'treatment', 'abdominal_pain', 'altered_smell', 'blisters_on_feet', 'brain_fog',
                 'chest_pain', 'chills_or_shivers', 'delirium', 'diarrhoea',
                 'diarrhoea_frequency', 'dizzy_light_headed', 'ear_ringing', 'earache',
                 'eye_soreness', 'fatigue', 'feeling_down', 'fever', 'hair_loss',
                 'headache', 'headache_frequency', 'hoarse_voice',
                 'irregular_heartbeat', 'loss_of_smell', 'nausea', 'persistent_cough', 'rash',
                 'red_welts_on_face_or_lips', 'runny_nose',
                 'shortness_of_breath', 'skin_burning', 'skipped_meals', 'sneezing',
                 'sore_throat', 'swollen_glands', 'typical_hayfever', 'unusual_muscle_pains', 'unusual_joint_pains',
                 'patient_id', 'created_at']

# Patient fields relevant to the analysis
patient_fields = ['id', 'year_of_birth', 'country_code', 'bmi', 'ethnicity', 'gender',
                  'is_pregnant', 'is_smoker', 'lsoa11cd', 'reported_by_another',
                  'has_asthma', 'has_eczema', 'has_hayfever', 'already_had_covid',
                  'has_cancer', 'has_diabetes', 'has_heart_disease', 'has_kidney_disease', 'has_lung_disease',
                  'does_chemotherapy', 'takes_immunosuppressants', 'contact_health_worker']


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


def get_assessments_tests_since_date(src_filename: str,
                                             dst_filename: str,
                                             start_date: datetime,
                                             save_name: str = 'data_tests_symptoms'):

    # Start an exetera session
    with Session() as s:
        # Open the dataset specified in read, specify where my data will be written
        src = s.open_dataset(src_filename, 'r', 'src')
        dst = s.open_dataset(dst_filename, 'w', 'dst')

        # Switch the order to look into the assessments table first
        # Then filter by date
        # Then merge with patients

        # Define a dataframe from original data (just work with patients' data)
        src_patients = src['patients']
        # Create a new destination dataframe
        d_patients = dst.create_dataframe('patients')
        print(len(src['assessments']['patient_id'].data[:]))

        # Copy the assessments and patient IDs into the new dataframe
        for f in patient_fields:
            df.copy(src_patients[f], d_patients, f)

        # Create and apply a filter on country code (limit to GB).
        # Note that this is stored as 'b' not string.
        filter = d_patients['country_code'].data[:] == b'GB'
        d_patients.apply_filter(filter)

        # Print the number of unique patients
        print(datetime.now(), len(np.unique(d_patients['id'].data[:])), ' number of unique patients found.')

        # Add in the patient comorbidities
        # Create an array
        nr_comorbidity = np.zeros(len(d_patients['has_diabetes'].data))

        # Loop over the comorbidities, if value is 2 in the data, place 1 in new array
        # or else add a 0 (check the schema)
        for f in ['has_diabetes', 'has_heart_disease', 'has_lung_disease', 'does_chemotherapy', 'has_kidney_disease',
                  'has_cancer', 'takes_immunosuppressants']:
            nr_comorbidity += np.where(d_patients[f].data[:] == 2, 1, 0)

        # Add and copy over the comorbidities
        d_patients.create_numeric('nr_comorbidity', 'int32')
        d_patients['nr_comorbidity'].data.write(nr_comorbidity)
        df.move(d_patients['id'], d_patients, 'id_patients')

        # Create a new dataframe for the assessments
        assessments = dst.create_dataframe('assessments')

        # For items in the list of symptoms, copy the data into the new dataframe
        for f in list_symptoms:
            df.copy(src['assessments'][f], assessments, f)

        # Rename the location series
        df.move(assessments['location'], assessments, 'hospitalisation')

        # Create a new dataframe
        d_assessments = dst.create_dataframe('d_assessments')

        # Merge the patients and assessments data
        # (inner merge, so no patients w/o assessments or vice versa)
        # Account for the different naming of the series to merge on
        # Account for the overlap in column names with the 'suffix'
        df.merge(d_patients, assessments, dest=d_assessments, how='inner', left_on='id_patients', right_on='patient_id',
                  right_suffix='_assessments')
        # Rename the created at series for the assessments to be obvious
        df.move(d_assessments['created_at'], d_assessments, 'created_at_assessments')
        # Print keys in the dataframe
        print('Keys in the assessments dataframe:')
        print(assessments.keys())

        # Filter for the time period we want to look at
        # Set the date that we want observations to start
        dt = start_date
        # Convert to a timestamp
        seconds = dt.timestamp()
        print('Timestamp for start date:', seconds)
        # Create a filter to pass over the existing data
        filters_date = (d_assessments['created_at_assessments'].data[:] >= seconds)
        # Apply the filter
        d_assessments.apply_filter(filters_date)


        # Merge in the test data
        print('Start merging tests')
        # Create a new dataframe
        v_tests = dst.create_dataframe('sympt_tests')

        # Merge with tests information
        # (inner... so only patients with tests remain)
        # Specify series from 'tests' to also include
        df.merge(d_assessments, src['tests'], dest=v_tests, how='inner', left_on='patient_id', right_on='patient_id',
                 right_fields=['created_at', 'date_taken_specific', 'mechanism', 'result', 'is_rapid_test'],
                 right_suffix='_test')
        # Print summary
        print('At time:', datetime.now(),
              'Unique patients:', len(np.unique(v_tests['patient_id'].data[:])),
              'with number of tests:', len(np.unique(v_tests['patient_id'].data[:])))

        # Update series name for clarity
        df.move(v_tests['date_taken_specific'], v_tests, 'date_test')

        # Sort by patient ID and then date of test
        test_doy = [datetime.fromtimestamp(i).timetuple().tm_yday for i in v_tests['date_test'].data[:]]
        v_tests.create_numeric('test_doy', 'int32').data.write(test_doy)
        v_tests.sort_values(by=['patient_id', 'test_doy'])

        # Save dataframe to csv
        save_df_to_csv(v_tests, 'data_tests.csv', list(v_tests.keys()))

        # Add vaccine data
        # Create a dataframe in the destination file
        p_vacc = dst.create_dataframe('p_vacc')
        # Merge the vaccine details with the assessments (note this includes flu vaccine info)
        df.merge(v_tests, src['vaccine_doses'], dest=p_vacc, how='left', left_on='patient_id', right_on='id',
                 right_fields=['id', 'sequence', 'date_taken_specific', 'brand', 'vaccine_type'])

        # Update series names for clarity
        df.move(p_vacc['date_taken_specific'], p_vacc, 'date_vaccine')
        # Save dataframe to csv
        save_df_to_csv(p_vacc, 'data_tests_vaccination.csv', list(p_vacc.keys()))



if __name__ == "__main__":
    srcfile = '/nvme0_mounts/nvme0lv01/exetera/recent/ds_20220523_full.hdf5'
    dstfile = 'data_con.hdf5'
    start_date = datetime(2022, 2, 1)
    get_assessments_tests_since_date(srcfile, dstfile, start_date, save_name=
                                             'data_con_v4.csv')
