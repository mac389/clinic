import json
import random

import numpy as np
import matplotlib.pyplot as plt
import Graphics as artist
import utils as tech

from pprint import pprint
from sklearn import linear_model, decomposition,preprocessing
from sklearn.feature_extraction import DictVectorizer
from scipy.stats import spearmanr

#Feature extraction vs feature selection

READ = 'rb'
WRITE ='wb'
TAB = '\t'
directory = json.load(open('directory.json',READ))


data = np.loadtxt(directory['working-set']['data'],delimiter = TAB, dtype=str)

with open(directory['working-set']['fields'],READ) as f:
	fields = map(format,f.readlines())

field_types = json.load(open(directory['field-types'],READ))

#Case the numerical features from str to int or float
patients = [{key:value if field_types[key] == 'str' else eval('%s(%s)'%(field_types[key],value)) 
			for key,value in patient.iteritems()} for patient in [dict(zip(fields,patient)) for patient in data]]

#Split into testing and training
#Randomize by shuffling and then splitting in half

random.shuffle(patients)
training = patients[:len(patients)/2]
testing = patients[len(patients)/2:]

#Get feature vectors zero-centered with mean variance.

#First let's extract features
vec = DictVectorizer()
x = vec.fit_transform(training).toarray()
y = vec.transform(testing).toarray()



labels = vec.get_feature_names()


heatmap = np.tril(cov(center(x))) + np.triu(cov(center(y)))
heatmap[np.diag_indices_from(heatmap)] = 1.0

#artist.dashboard(tech.eigendecomp(heatmap,numpc=3),labels=map(format,labels),ed=True)


#Variables with |r| >0.20 are strongly correlted with EVD score

idx = np.where(np.absolute(heatmap[:,2])>0.20)[0]
good_labels = [labels[i] for i in idx]
good_labels += ['Location']

'''
with open('../Data/for-mvr.fields',WRITE) as f:
	for item in good_labels:
		print>>f,item

with open('../Data/for-mvr.data',WRITE) as f:
	json.dump([{key:value for key,value in patient.iteritems() if key in good_labels} 
									for patient in training], f)

with open('../Data/evd-scores.data',WRITE) as f:
	json.dump([patient['EVD score'] for patient in training],f)

with open('../Data/for-mvr-evaluation.data',WRITE) as f:
	json.dump([{key:value for key,value in patient.iteritems() if key in good_labels} 
									for patient in testing], f)

with open('../Data/evd-scores-evaluation.data',WRITE) as f:
	json.dump([patient['EVD score'] for patient in testing],f)
'''

''' Projection of correlations onto EVD score
fig = plt.figure()
coeffs = fig.add_subplot(111)
coeffs.barh(range(len(sorted)),sorted, edgecolor='k',
		color = ['r' if x<0 else 'g' for x in sorted],
		linewidth=1)
artist.adjust_spines(coeffs)
coeffs.axvline(x=0,color='k',linestyle='--',linewidth=2)
coeffs.axvline(x=-.23,color='r',linestyle='--',linewidth=1)
coeffs.axvline(x=.23,color='r',linestyle='--',linewidth=1)
coeffs.set_xlabel(r'\Large \textbf{Correlation with EVD accuracy}')
coeffs.set_yticks(range(len(labels)))
coeffs.set_yticklabels(map(artist.format,[labels[i] for i in idx]))
fig.tight_layout()
plt.show()
'''
