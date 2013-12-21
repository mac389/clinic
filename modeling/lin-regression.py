import json
import csv

import Graphics as artist
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, svm
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from pprint import pprint
from matplotlib import rcParams

from random import sample

rcParams['text.usetex'] = True

READ = 'rU'
WRITE = 'wb'
directory = json.load(open('directory.json',READ))

def convert(item):
	item = item.strip()
	if item == 'M' or item == 'vps' or item == 'ICU' or item == 'LF' or item == 'L':
		return -1
	elif item == 'F' or item == 'evd' or item == 'OR' or item == 'RF' or item == 'R':
		return 1
	elif item == 'om' or item == 'ED' or item == 'LT':
		return 0
	elif item == 'LO' or item == 'AG':
		return -2
	elif item == 'RO':
		return 2
	else:
		return item

#Could all this be done with a GNUplot script?
with open('../Data/variables',READ) as f:
	vois = [x.rstrip('\t\n') for x in f.readlines()]

translation = json.load(open('../Data/better-names.json',READ))

cols = set(range(35))
bad_cols = set([0,1,2,12,13,29,30])
good_cols = list(cols-bad_cols)

conversion = dict(zip(range(35),vois))

labels = [translation[conversion[col]] for col in good_cols if col !=9]
with open(directory['data'],READ) as fid:
	reader = csv.reader(fid)
	reader.next()

	data =np.array(filter(lambda row: '' not in row and 'NA' not in row and '?' not in row,
			[[convert(row[i]) for i in good_cols] for row in reader])).astype(float)
	loc = data[:,9]
	tumor = data[:,4]
	data = (data- data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
	y = data[:,7]
	x = np.delete(data,7,1)

#Dimension reduction
pca = decomposition.PCA(n_components=5)
logistic = linear_model.LogisticRegression()
pipe = Pipeline(steps=([('pca',pca),('logistic',logistic)]))

pca.fit(x)

mat = pca.components_.transpose()*pca.explained_variance_ratio_
idxs = np.argsort(mat[:,0])
#artist.heatmap(mat[idxs,:], 
#	xlabel='Principal component (scaled by eigenvalue)',
#	yticklabels=[labels[i] for i in idxs], pc_cutoff = 5)

n_components = [5, 10, 15]
Cs = np.logspace(-4, 4, 3)
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(x,y)

artist.heatmap(pca.transform(x), ylabel='Patients', xlabel='Eigenvectors')

z = pca.transform(x)
fi = plt.figure()
a = fi.add_subplot(111)
a.scatter(z[:,0],z[:,1],c=['r' if x == -1 else 'g' for x in loc], 
		s=[50 if x==1 else 20 for x in tumor],alpha = [0.3+k for k in y] )
artist.adjust_spines(a)
a.set_xlabel(r'\Large \textbf{PC1}')
a.set_ylabel(r'\Large \textbf{PC2}')
plt.show()

'''
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
cax = ax2.imshow(logistic.coef_.transpose(),interpolation='nearest',aspect='auto')
artist.adjust_spines(ax2)
ax2.set_xticks(range(3))
ax2.set_xticklabels(range(1,4))
ax2.set_yticks(range(len(good_cols)))
#ax.set_yticklabels()
plt.colorbar(cax)
plt.tight_layout()
'''
'''
fig = plt.figure(1,figsize=(4,3))
ax = fig.add_subplot(111)
ax.plot(pca.explained_variance_,linewidth=2,color='k')
ax.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
           linestyle=':', label='n_components chosen')
artist.adjust_spines(ax)
ax.set_xlabel(r'\Large \textbf{Component}')
ax.set_ylabel(r'\Large \textbf{Explained variance}')
plt.legend(prop=dict(size=12))
plt.tight_layout()
'''
plt.show()


'''
c,resid,rank,sigma = np.linalg.lstsq(x.transpose(),y.transpose())

np.savetxt('../Data/x.tsv',x, fmt='%.02f', delimiter = '\t')
np.savetxt('../Data/y.tsv',y, fmt = '%.02f', delimiter = '\t')

coeffs = sorted(zip(c,good_cols),key=lambda entry:abs(entry[0]))[::-1]
coeffs = [(coeff[0],conversion[coeff[1]]) for coeff in coeffs]

format = lambda cf: r'\textbf{%s}'%cf.capitalize()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.barh(range(len(coeffs)),[abs(coeff[0]) for coeff in coeffs][::-1],
		color=['k' if coeff[0]>0 else 'r' for coeff in coeffs], xerr=sigma[::-1],
		ecolor='k')
artist.adjust_spines(ax)
ax.set_xlabel(format('Weight'))
plt.yticks(range(len(coeffs)),map(format,[coeff[1] for coeff in coeffs])[::-1])
plt.tight_layout()
plt.show()
'''