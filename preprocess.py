import pandas as pd

raw_data = pd.read_csv('Hospital.csv')


names = {
    'Hospital County':'county',
    'Facility Id':'facility',
    'Age Group':'age',
    'Gender':'gender',
    'Race':'race',
    'Ethnicity':'ethnicity',
    'Length of Stay':'stay',
    'Admit Day of Week':'admit_day',
    'Type of Admission':'admit_type',
    'Patient Disposition':'disposition',
    'CCS Diagnosis Code':'diagnosis',
    'CCS Procedure Code':'procedure',
    'APR DRG Code':'DRG',
    'APR MDC Code':'MDC',
    'APR Severity of Illness Code':'severity',
    'APR Risk of Mortality':'risk',
    'APR Medical Surgical Description':'method',
    'Emergency Department Indicator':'emg' 
}


payment_methods = {
    'Payment Typology 1':'payment1',
    'Payment Typology 2':'payment2',
    'Payment Typology 3':'payment3',
}


raw_data.rename(columns=names, inplace=True)

# select relevant variables
data = raw_data[sorted(names.values())]
data.shape


# data clearning and data transformation
data = data.dropna()

data['facility'] = data['facility'].astype(int)
data = data[data['gender'] != 'U']
data = data[data['race'] != 'Unknown']
data['diagnosis'] = data['diagnosis'].astype(str)
data['procedure'] = data['procedure'].astype(str)
data['DRG'] = data['DRG'].astype(str)
data['MDC'] = data['MDC'].astype(str)
data['severity'] = data['severity'].astype(str)

data.to_csv('cleaned_data.csv')

