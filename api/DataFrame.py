import csv, json, os, random

import numpy as np
import matplotlib.pyplot as plt

from api import utils as tech
from visualization import Graphics as artist
from pprint import pprint
from sklearn import linear_model, decomposition,preprocessing
from sklearn.feature_extraction import DictVectorizer
from scipy.stats import spearmanr

READ = 'rU'
WRITE = 'wb'
TAB = '\t'

#Must make filters to automatically exclude certain fields that have values for too few records.

directory = json.load(open('./data/directory.json',READ))

class DataFrame(object):

	def __init__(self,filename):

		format_line = lambda line: line.rstrip('\n')  
		directory['data'] = filename

		if not all(map(os.path.isfile, directory['working-set'])):
			self.db = list(csv.DictReader(open(directory['data'],READ)))
			self.excluded = map(format_line,open(directory['excluded'],READ).readlines())
			self.missing_values = map(format_line,open(directory['missing-values'],READ).readlines())
			
			self.db=[{key:value for key,value in patient.iteritems() 
								if key not in self.excluded} 
								for patient in self.db]

			#self.db = filter(lambda patient: patient['Ventriculostomy']=='evd',self.db)

			self.db = filter(lambda patient: not any([value in self.missing_values 
												for value in patient.values()]),self.db)

			for patient in self.db:
				del patient['Ventriculostomy']

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

			np.savetxt(directory['working-set']['data'],self.data, fmt='%s',delimiter=TAB)
			with open(directory['working-set']['fields'],WRITE) as f:
				for item in self.keys:
					print>>f,item
