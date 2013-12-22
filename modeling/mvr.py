import json,random

import numpy as np
import api.utils as tech

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from visualization import visualization
from pprint import pprint
from scipy.stats import spearmanr


READ = 'rU'
WRITE = 'wb'
TAB = '\t'
directory = json.load(open('./data/directory.json',READ))

class MVR(object):

	def __init__(self,dataframe):
		#Error check later, for now assume that this is a data frame
		self.dataframe = dataframe

		with open(directory['working-set']['fields'],READ) as f:
			self.fields = map(tech.format,f.readlines())
		
		field_types = json.load(open(directory['field-types'],READ))

		#Cast the numerical features from str to int or float
		self.patients = [{key:value if field_types[tech.format(key)] == 'str' else eval('%s(%s)'%(field_types[tech.format(key)],value)) 
						for key,value in patient.iteritems()} 
						for patient in self.dataframe.db]

		#Save fully processed array. 

		with open('./data/fully-processed.json',WRITE) as f:
			json.dump(self.patients,f)

		directory['fully-processed'] = './data/fully-processed.json'

		#Make covariance matrix to identify variables of interest

		#Split into testing and training
		#Randomize by shuffling and then splitting in half

		random.shuffle(self.patients) #Make this a parameter so can shuffle many times, store indices
		self.training = self.patients[:len(self.patients)/2]
		self.testing = self.patients[len(self.patients)/2:]

		#Extract features
		self.vec = DictVectorizer()
		self.x = self.vec.fit_transform(self.patients).toarray()

		covariance,prob = spearmanr(self.x)
		self.cov_matrix = covariance*(prob<0.05)
		np.savetxt('./data/covariance-matrix.tsv',self.cov_matrix,fmt='%.04f',delimiter=TAB)
		directory['covariance'] = {}
		directory['covariance']['data'] = './data/covariance-matrix.tsv'

		with open('./data/covariance-matrix.fields',WRITE) as f:
			for item in self.vec.get_feature_names():
				print>>f,item
		directory['covariance']['labels'] = './data/covariance-matrix.fields'
		
		#visualization.covariance(self.cov_matrix,self.vec.get_feature_names(),show=True,ml=False)

		#Extract the fields that are significantly correlated with at least one of the EVD scores. 
		self.labels = self.vec.get_feature_names()

		self.idx = np.where(self.cov_matrix[2:5,:]>0)[1]
		self.good_labels = [self.labels[i] for i in self.idx if 'EVD' not in self.labels[i]]
		pprint(self.good_labels)
		
		with open('../Data/for-mvr.fields',WRITE) as f:
			for item in self.good_labels:
				print>>f,item

		with open('../Data/for-mvr.data',WRITE) as f:
			json.dump([{key:value for key,value in patient.iteritems() if key in self.good_labels} 
											for patient in self.training], f)

		with open('../Data/evd-scores.data',WRITE) as f:
			json.dump([patient['EVD score'] for patient in self.training],f)

		with open('../Data/for-mvr-evaluation.data',WRITE) as f:
			json.dump([{key:value for key,value in patient.iteritems() if key in self.good_labels} 
											for patient in self.testing], f)

		with open('../Data/evd-scores-evaluation.data',WRITE) as f:
			json.dump([patient['EVD score'] for patient in self.testing],f)
		

		#Build logistic model

		self.train_data = json.load(open(directory['mvr']['data'],READ))
		self.train_outcome = json.load(open(directory['evd-score'],READ))

		self.extractor = DictVectorizer()
		self.train_data_array = self.extractor.fit_transform(self.train_data).toarray()

		self.model = LogisticRegression()
		self.model.fit(self.train_data_array,self.train_outcome)

		#Evaluate logistic model

		self.test_data = json.load(open(directory['evaluation']['data'],READ))
		self.test_outcome = json.load(open(directory['evaluation']['evd-scores'],READ))
		self.values = range(1,4)

		self.test_predictions = np.array(self.model.predict(self.extractor.transform(self.test_data).toarray())).transpose()

		print zip(self.test_outcome,self.test_predictions)

		#print self.model.decision_function(self.extractor.transform(self.test_data).toarray())
		#pprint(zip(self.model.coef_.transpose(),self.extractor.get_feature_names()))

		#visualization.mvr_coefficients(self.model,self.extractor.get_feature_names(),show=True)		

		self.test_outcome = np.array(self.test_outcome).astype(int)
		self.test_predictions = np.array(self.test_predictions).astype(int)

		self.contingency_table = np.array([[sum((self.test_outcome==i)*(self.test_predictions==j)) 
			for i in self.values] for j in self.values])

		pprint(self.contingency_table)


		accuracy = lambda data: data[np.diag_indices_from(data)].sum()/float(data.sum())
		print accuracy(self.contingency_table)
#tests = zip([sensitivity,specificity,ppv,npv,accuracy],['Sensitivity','Specificity','NPV','PPV','Accuracy'])

#summarize(contingency_table)