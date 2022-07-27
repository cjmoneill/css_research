# css_research
Code for covid symptom study research project (Healthcare Technologies MRes)

Script descriptions:

# exetera_script.py

- Takes a CSS hdf5 file and a start date (to limit observations to a specific period) inputs
- Creates a dataframe of assessments provided since the start date
- Filters these assessments for country of interest (GB)
- Adds patient comorbidities to the assessments
- Adds reported symptoms for each assessment
- Adds vaccination status to each patient
- Adds test data for each patient
- Saves resulting dataframe to csv

# main.py

- Takes a csv created in `exetera_script.py` as an input
- Prints basic information (total patients, assessments, patient tests etc)
- Removes assessments/patients according to inc/exclusion criteria:
  - Patients within invalid BMI
  - Patients whose assessments have been logged by a proxy
  - Patients under 18 (in 2022)
- Provides the number of assessments and tests following above exclusions
- Applies further inclusion criteria:
  - Patients who provided at least 2 assessments in at least 2/3 of the weeks in the period being evaluated
- Adds a column identifying patient nation of residence (England/Scotland/Wales/Northern Ireland)
- Creates and saves patient/assessments dataframes for each nation (patients meeting inclusion criteria only)
- Prints summary information for each nation
- Compares distribution frequency of tests per patient, before/after a policy change
- Prints demographic information for a dataframe (gender, BMI, chemotherapy, asthma etc)

# merge_contact_worker.py

- Takes a patient/assessments dataframe and a CSS patients export as inputs
- Extracts the contact_health_worker field for all patients in the CSS export
- Merges with a patient/assessments dataframe
- Saves the resulting dataframe

# stratifying_test_train_samples.py

- Takes a patients/assessments dataframe as input
- Calls functions to create a dataframe which includes classification targets (whether or not the patient tested less after policy changes)
- Calculates the desirable number of patients of each class to include in a test/train split
- Has dictionary of the ideal number of patients in each class for the test set (needs updating when input is changes)
- Calls `create_stratified_test_sample function to create and save a train and test sets which meet the criteria set out in the dictionary above

# model_evaluation_stratified.py

- Takes stratified test and train datasets (csv) as inputs
- Tidies up `health_worker_status` values by converting nans to 0
- Prints demographic information
- Splits dataframes into feature and target vectors (X_train, y_train, X_test, y_test)
- Scales feature vectors
- Creates a random forest classfier, using grid search to find best hyperparameters
- Evaluates performance on train and test data
- Prints ROC curve and score
- Calculates Youden's J statistic
- Visualises comparative performance for hyperparameters
