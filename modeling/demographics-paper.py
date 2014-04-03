import csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn import decomposition, linear_model
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr

from pprint import pprint
from progress.bar import Bar

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def cov(data, labels,convert_types):

	bar = Bar('Calculating covariance matrix',max=(data.shape[0]*(data.shape[0]-1)/2))
	answer = np.zeros((data.shape[0],data.shape[0]))
	for row in xrange(data.shape[0]):
		for col in xrange(row):
			association = spearmanr(data[row,:],data[col,:])[0] if convert_types[labels[row].split('=')[0]] == float else normalized_odds_ratio(row,col)
			
			answer[row][col] = association
			bar.next()
	bar.finish()
	answer += answer.transpose()
	answer[np.diag_indices_from(answer)] = 1.0

	return answer
	'''
	return np.array([[spearmanr(data[i,:],data[j,:])[0] 
				if convert_types[labels[i].split('=')[0]] == float else normalized_odds_ratio(i,j)
					for i in xrange(data.shape[0])] for j in xrange(data.shape[0])])
	'''
def normalized_odds_ratio(i,j):
	#Assuming that i and j are row indices

	row_values = np.unique(data[i,:])
	col_values = np.unique(data[j,:])
	contingency_table = np.array([[sum((data[i,:]==row_val)*(data[j,:]==col_val)) 
		for row_val in row_values] for col_val in col_values])


	diag = np.diag(contingency_table).sum()
	rest = contingency_table.sum()-diag.sum()
	return (diag - rest)/float(diag+rest)

#--Extract features
with open('../data/th-demographics.csv','rU') as f:
	reader = csv.DictReader(f)
	measurements = list(reader)

#convert anything that can be a number into a number
variable_types = {}
for i,measurement in enumerate(measurements):
	for label,datum in measurements[i].iteritems():
		if is_number(datum) and label not in ['Session','Session ID ']:
			measurements[i][label] = float(datum)
			variable_types[label] = float
		else:
			variable_types[label] = str

vec = DictVectorizer()
data = vec.fit_transform(measurements).toarray().transpose()

r = cov(data,vec.get_feature_names(),variable_types)
with open('../data/demo-feature-names','wb') as f:
	for name in vec.get_feature_names():
		print>>f, name


np.savetxt('../data/cov-matrix_.tsv',r, fmt='%.04f',delimiter='\t')
'''
fig = plt.figure()
ax = fig.add_suplot(111)
ax.imshow(f,interpolation='nearest',aspect=auto)
plt.show()
'''