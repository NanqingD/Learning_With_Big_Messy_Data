import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('cleaned_data.csv')

names = {
    'Hospital County':'county',
    'Facility Id':'facility',
    'Age Group':'age',
    'Gender':'gender',
    'Race':'race',
    'Ethnicity':'ethnicity',
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


def plotBar(varName, data):
	global standard_deviation
	fig = plt.figure()
    subdata = data[varName]
    freq = dict(subdata.value_counts())
    expired = {}
    percentage = {}
    for k in sorted(freq.keys()):
        expired[k] = len(subdata[(data[varName] == k) & (data['disposition'] == 'Expired')])
        percentage[k] = expired[k] * 1.0 / freq[k]
    percentages = np.array(percentage.values())
    standard_deviation[varName] = np.std(percentages)
    plt.bar((range(1,len(percentage)+1)), percentage.values(), align='center')
    plt.title('%s' %varName)   
    plt.savefig('%s.png' %varName)
    plt.close(fig)
    print '%s finished' %varName

standard_deviation = {}

for varName in sorted(names.values()):
    if varName == 'disposition':
    	pass
    plotBar(varName, data)

print standard_deviation

# admit_day
def plotAdmitDay(data):
	varName = 'admit_day'
	fig = plt.figure()
	subdata = data[varName]
	freq = dict(subdata.value_counts())
	expired = []
	percentage = []
	week = ['SUN', 'MON', 'SAT', 'TUE', 'WED', 'THU', 'FRI']
	for i,k in enumerate(week):
	    expired[k] = len(subdata[(data[varName] == k) & (data['disposition'] == 'Expired')])
	    percentage[i] = expired[k] * 1.0 / freq[k]
	plt.bar((range(1,len(percentage)+1)), percentage, align='center')
	plt.title('Mortality vs Each Admitted Day of A Week')
	plt.xlim(0, 8)
	plt.xlabel('admit day')
	plt.ylabel('percentage')
	plt.xticks(range(0,len(week)),week,rotation=45)  
	plt.savefig('%s.png' %varName)
	plt.close(fig)
	print '%s finished' %varName


# MDC
def plotMDC(data):
	varName = 'MDC'
	fig = plt.figure()
	subdata = data[varName]
	freq = dict(subdata.value_counts())
	expired = {}
	keys = sorted(map(int,freq.keys()))
	percentage = [0]*len(keys)
	for i,k in enumerate(keys):
	    k = str(k)
	    expired[k] = len(subdata[(data[varName] == k) & (data['disposition'] == 'Expired')])
	    percentage[i] = expired[k] * 1.0 / freq[k]
	plt.bar((range(1,len(keys)+1)), percentage, align='center')
	plt.title('Mortality vs All Patient Refined \n Major Diagnostic Category(APR MDC) Description')
	plt.xlim(-1, len(keys)+2)
	plt.xlabel('APR MDC Description')
	plt.ylabel('Percentage')
	plt.xticks(range(1,len(keys)+1),keys,rotation=90)  
	plt.savefig('%s.png' %varName)
	plt.close(fig)