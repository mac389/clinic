import csv
import json

import numpy as np

from pprint import pprint

READ = 'rU'
WRITE = 'wb'

directory = json.load(open('./data/directory.json',READ))

class DataFrame(object):

	def __init__(self,filename):

		format_line = lambda line: line.rstrip('\n')  
		directory['data'] = filename

		self.db = list(csv.DictReader(open(directory['data'],READ)))
		self.excluded = map(format_line,open(directory['excluded'],READ).readlines())
		self.missing_values = map(format_line,open(directory['missing-values'],READ).readlines())
		
		self.db=[{key:value for key,value in patient.iteritems() if key not in self.excluded} 
							for patient in self.db]

		self.db = filter(lambda patient: not any([value in self.missing_values 
											for value in patient.values()]),self.db)

		self.keys = self.db[0].keys()
		self.data = np.array([[patient[field].strip() for field in self.keys] 
													  for patient in self.db]).astype(str)

		self.converter = {field:i for i,field in enumerate(self.keys)}


		'''
		Excluded fields
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