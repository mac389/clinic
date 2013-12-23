import json,random,os

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

	def __init__(self,dataframe,suffix='',basedir=''):

		self.basedir = '../Data/' if basedir == '' else '../Data/%s/'%basedir

		#Error check later, for now assume that this is a data frame
		self.dataframe = dataframe
		self.suffix = suffix

		with open(directory['working-set']['fields'],READ) as f:
			self.fields = map(tech.format,f.readlines())
		
		self.field_types = json.load(open(directory['field-types'],READ))

		#Cast the numerical features from str to int or float
		self.patients = [{key:value if self.field_types[tech.format(key)] == 'str' else eval('%s(%s)'%(self.field_types[tech.format(key)],value)) 
						for key,value in patient.iteritems()} 
						for patient in self.dataframe.db]

		#Save fully processed array. 

		with open('%sfully-processed-%s.json'%(self.basedir,self.suffix),WRITE) as f:
			json.dump(self.patients,f)

		#Make covariance matrix to identify variables of interest

		#Split into testing and training
		#Randomize by shuffling and then splitting in half

		self.idx = range(len(self.patients))
		random.shuffle(self.idx) 
		self.training = [self.patients[i] for i in self.idx[:len(self.idx)/2]]
		self.testing = [self.patients[i] for i in self.idx[len(self.idx)/2:]]

		with open('%siteration-%s.indices'%(self.basedir,self.suffix),WRITE) as f:
			print>>f,self.idx

		#Extract features
		self.vec = DictVectorizer()
		self.x = self.vec.fit_transform(self.patients).toarray()

		if not os.path.isfile('./data/covariance-matrix.tsv'):

			covariance = self.cov()
			'''
			covariance,prob = spearmanr(self.x)
			self.cov_matrix = covariance*(prob<0.05)
			'''
			np.savetxt('./data/covariance-matrix.tsv',self.cov_matrix,fmt='%.04f',delimiter=TAB)
			directory['covariance'] = {}
			directory['covariance']['data'] = './data/covariance-matrix.tsv'

			with open('%scovariance-matrix.fields'%self.basedir,WRITE) as f:
				for item in self.vec.get_feature_names():
					print>>f,item
			directory['covariance']['labels'] = './data/covariance-matrix.fields'
		
		#visualization.covariance(self.cov_matrix,self.vec.get_feature_names(),show=True,ml=False)

		#Extract the fields that are significantly correlated with at least one of the EVD scores. 
		self.labels = self.vec.get_feature_names()

		self.idx = np.where(self.cov_matrix[2:5,:]>0)[1]
		self.good_labels = [self.labels[i] for i in self.idx if 'EVD' not in self.labels[i]]

		with open('%sfor-mvr-%s.fields'%(self.basedir,self.suffix),WRITE) as f:
			for item in self.good_labels:
				print>>f,item

		with open('%sfor-mvr-%s.data'%(self.basedir,self.suffix),WRITE) as f:
			json.dump([{key:value for key,value in patient.iteritems() if key in self.good_labels} 
											for patient in self.training], f)

		with open('%sevd-scores-%s.data'%(self.basedir,self.suffix),WRITE) as f:
			json.dump([patient['EVD score'] for patient in self.training],f)

		with open('%sfor-mvr-evaluation-%s.data'%(self.basedir,self.suffix),WRITE) as f:
			json.dump([{key:value for key,value in patient.iteritems() if key in self.good_labels} 
											for patient in self.testing], f)

		with open('%sevd-scores-evaluation-%s.data'%(self.basedir,self.suffix),WRITE) as f:
			json.dump([patient['EVD score'] for patient in self.testing],f)
		
		#Build logistic model

		self.train_data = json.load(open('%sfor-mvr-%s.data'%(self.basedir,self.suffix),READ))
		self.train_outcome = json.load(open('%sevd-scores-%s.data'%(self.basedir,self.suffix),READ))

		self.extractor = DictVectorizer()
		self.train_data_array = self.extractor.fit_transform(self.train_data).toarray()

		self.model = LogisticRegression()
		self.model.fit(self.train_data_array,self.train_outcome)

		#Evaluate logistic model

		self.test_data = json.load(open('%sfor-mvr-evaluation-%s.data'%(self.basedir,self.suffix),READ))
		self.test_outcome = json.load(open('%sevd-scores-evaluation-%s.data'%(self.basedir,self.suffix),READ))
		self.values = range(1,4)

		self.test_predictions = np.array(self.model.predict(self.extractor.transform(self.test_data).toarray())).transpose()

		visualization.mvr_coefficients(self.model,self.extractor.get_feature_names(),
			savename='%smvr-coefficients-%s'%(self.basedir,self.suffix))	

		self.test_outcome = np.array(self.test_outcome).astype(int)
		self.test_predictions = np.array(self.test_predictions).astype(int)

		self.contingency_table = np.array([[sum((self.test_outcome==i)*(self.test_predictions==j)) 
			for i in self.values] for j in self.values])

		accuracy = lambda data: data[np.diag_indices_from(data)].sum()/float(data.sum())

		self.evaluation = tuple((self.contingency_table,accuracy(self.contingency_table)))
		#tests = zip([sensitivity,specificity,ppv,npv,accuracy],['Sensitivity','Specificity','NPV','PPV','Accuracy'])
	
	def cov(self):
		return np.array([[spearmanr(self.x[i,:],self.x[j,:]) 
						if self.field_types[i] == "float" or self.field_types[j] == "float"
						else self.normalized_odds_ratio(i,j)
						for i in xrange(self.x.shape[0])] 
						for j in xrange(self.x.shape[0])])

	def normalized_odds_ratio(self,i,j):
		#Assuming that i and j are row indices

		row_values = np.unique(self.x[i,:])
		col_values = np.unique(self.x[j,:])
		
		return (diag + off_diag)/float(diag-off_diag)