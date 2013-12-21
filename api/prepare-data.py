import csv
import json

import numpy as np

from pprint import pprint

READ = 'rU'
WRITE = 'wb'

directory = json.load(open('directory.json',READ))
db = list(csv.DictReader(open(directory['data'],READ)))
excluded = ['Guidance','MRN','date','Type of Guidance','shortest distance from midline',
	'shortest distance from coronal suture','CNS_Diagnosis']

'''Excluded fields
	; Guidance
	; MRN
	; Date
	; Sex
	; Type of Guidance
	; Shortest distance from midline
	; Shortest distance from coronal suture
	; CNS Diagnosis 

	These fields were excluded because too few patients had them (like less than 50%)
	Any row exluded that did not have all values

'''

missing_vaules = ['NA','','?']
print len(db)
db=[{key:value for key,value in patient.iteritems() if key not in excluded} for patient in db]
db = filter(lambda patient: not any([value in missing_vaules for value in patient.values()]),db)
print len(db)

keys = db[0].keys()
data = np.array([[patient[field].strip() for field in keys] for patient in db]).astype(str)
pprint(data)

converter = {field:i for i,field in enumerate(keys)}


np.savetxt('../Data/working-data-set.data',data, fmt='%s',delimiter='\t')
with open('../Data/working-data-set.fields',WRITE) as f:
	for item in keys:
		print>>f,item
