import csv,sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn import decomposition, linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

from scipy.stats import spearmanr

from pprint import pprint
from progress.bar import Bar

from visualization import visualization

filename = sys.argv[1]
READ = 'rU'

measurements = list(csv.DictReader(open(filename,READ)))

numeric = ['total length (stripe length)', 
'Right Caudate to Septum Distance',
'Line through caudate heads to inner tables', 'angle with AP line', 
'shortest distance from midline', 'tangent with vertical (angle)', 
'Bicaudate index', 'Left Caudate to Septum Distance ', 
'mean burr hole diameter', 'mean thickness', 
'shortest distance from coronal suture', 'slice thickness', 
'Bicaudate distance', 'tangent with horizontal', 'length in ventricle', 
'angle with horizontal', 'age', 'burr hole diameter. Ext', 
'burr hole diameter 2. Int', 'length in brain (LP)', 'skull thickness 1',
 'skull thickness 2','midline shift']

for i,measurement in enumerate(measurements):
	for field in numeric:
		tmp =measurements[i][field].strip() 
		tmp = float(tmp) if tmp not in ['','NA','?'] else np.nan 
		measurements[i][field] = tmp

	#recode evd score
	tmp = measurements[i]['EVD score']
	if tmp not in ['','NA','?']:
		measurements[i]['EVD score'] = 1 if float(tmp)>2 else 0
	else:
		print i
		print tmp
		measurements[i]['EVD score'] = np.nan
vec = DictVectorizer()

y = [x['EVD score'] for x in measurements]

verboten = ['Guidance','MRN','date','Type of Guidance','shortest distance from midline',
'shortest distance from coronal suture' ,'CNS_Diagnosis','EVD score']

for x in measurements:
	for field in verboten:
		del x[field]

x = vec.fit_transform(measurements)
names = [name for name in vec.get_feature_names() if name !='EVD score']
imp = Imputer(axis=0,copy=True,missing_values=np.nan,strategy='mean')
imp.fit(x)
x_imp = imp.transform(x)
regress = linear_model.LogisticRegression()

z = regress.fit(x_imp,y)
visualization.coefficients(z,names,show=True, title='EVD, VPS, OM')