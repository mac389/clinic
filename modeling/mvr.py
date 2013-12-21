import json

import numpy as np

from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

sensitivity = lambda data: data[0,0]/float(data[:,0].sum())
specificity = lambda data: data[1,1]/float(data[:,1].sum())

ppv = lambda data: data[0,0]/float(data[0,:].sum())
npv = lambda data: data[1,1]/float(data[1,:].sum())

accuracy = lambda data: data[np.diag_indices_from(data)].sum()/float(data.sum())

tests = zip([sensitivity,specificity,ppv,npv,accuracy],['Sensitivity','Specificity','NPV','PPV','Accuracy'])

def summarize(data):
	print '-----------'
	for test,label in tests:
		print '%s : %.03f'%(label,test(data))
	print '-----------'

READ = 'rb'
WRITE = 'wb'
TAB = '\t'
directory = json.load(open('directory.json',READ))

predictors = json.load(open(directory['mvr']['data'],READ))
EVD = json.load(open(directory['evd-score'],READ))

extractor = DictVectorizer()
x = extractor.fit_transform(predictors).toarray()

model = LogisticRegression()
model.fit(x,EVD)

predictions = np.array(model.predict(x))
EVD = np.array(EVD)
values = np.unique(predictions)

contingency_table = np.array([[sum((EVD==i)*(predictions==j)) for i in values] for j in values])

pprint(contingency_table)
summarize(contingency_table)

#Evaluate on testing data
test_predictors = json.load(open(directory['evaluation']['data'],READ))
test_scores = json.load(open(directory['evaluation']['evd-scores'],READ))


test_predictions = np.array(model.predict(extractor.transform(test_predictors).toarray())).transpose()
test_contingency_table = np.array([[sum((test_scores==i)*(test_predictions==j)) for i in values] for j in values]).transpose()
summarize(test_contingency_table)

print ''
print ''
print model.decision_function(extractor.transform(test_predictors).toarray())
pprint(zip(model.coef_.transpose(),extractor.get_feature_names()))

import Graphics as artist
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(np.delete(model.coef_.transpose(),2,0),interpolation='nearest',aspect='auto')
artist.adjust_spines(ax)
ax.set_yticks(range(len(extractor.get_feature_names())-1))
ax.set_yticklabels(map(artist.format,[name for name in extractor.get_feature_names() if 'EVD' not in name]))
ax.set_xticks(range(3))
ax.set_xticklabels(map(artist.format,range(1,4)))
ax.set_xlabel(artist.format('Placement grade'))
plt.colorbar(cax)
plt.tight_layout()
plt.show()