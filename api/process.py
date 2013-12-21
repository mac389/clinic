import csv
import json
import argparse

import numpy as np
import Graphics as artist

from pprint import pprint
from itertools import combinations
from sklearn.linear_model import LogisticRegression

READ = 'rU'
WRITE = 'wb'

directory = json.load(open('directory.json',READ))
db = list(csv.DictReader(open(directory['data'],READ)))

#Could all this be done with a GNUplot script?
with open('../Data/variables',READ) as f:
	vois = [x.rstrip('\t\n') for x in f.readlines()]

'''
parser = argparse.ArgumentParser(description='Analyze ventriculostomy data')
parser.add_argument('vars',nargs='+', help='List of variables of interest. Default is all variables.',
		default = all_vois)
args = parser.parse_args()
vois = args.vars #Variables of interest
print vois
'''
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

def extractData(voi):
	if type(voi) != list:
		return np.array(filter(lambda entry: entry != 'NA' and entry != '' and entry !='?',
							[convert(item[voi]) for item in db])).astype(float)
	else:
		rec = np.array([[convert(item[v]) for item in db] for v in voi])
		return rec

def bad_indices(voi):
	rec = [item[voi] for item in db]
	idx =  np.where(np.char.strip(rec) == 'NA')[0]
	return idx

'''	
for one,two in combinations(vois,2):
	print one,two
	bad_idx = list(bad_indices(one)) + list(bad_indices(two))
	good_idx = list(set(range(min(len(extractData(one)),len(extractData(two))))) - set(bad_idx))

	#Find indices where neither has an invalid entry
	artist.scatter(extractData(one)[good_idx],extractData(two)[good_idx],xlabel=one,ylabel=two, 
		savename='../Figures/%s-%s'%(one,two))


for voi in vois: 
	#Calculate distribution of item in database
	#May have to add a spell-checker
	#Sloppy exclusion
	artist.hist(extractData(voi),xlabel=voi, savename='../Figures/%s'%voi)
'''

import matplotlib.pyplot as plt
import Graphics as artist

from scipy.stats import ks_2samp, scoreatpercentile
'''
iqr = lambda arr: 0.5*(scoreatpercentile(arr,75)-scoreatpercentile(arr,25))

gender = extractData('sex')
age = extractData('age')

m = age[gender==-1]
f = age[gender==1]

#for propensity score
age_gender = np.array(filter(lambda row: 'NA' not in row,extractData(['age','sex','Ventriculostomy']).transpose())).astype(int)
cls = LogisticRegression()
cls.fit(age_gender[:,:2],age_gender[:,2])
print cls.coef_,'kkk'
pprint(ks_2samp(m,f))

print np.median(m),iqr(m)

print np.median(f),iqr(f)

format = lambda str: r'\Large \textbf{%s}'%str
'''

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(f,histtype='step',color='r',alpha=0.6,label=format('Female'),cumulative=True)
plt.hold(True)
ax.hist(m,histtype='step',color='b',alpha=0.7,label=format('Male'), cumulative=True)
artist.adjust_spines(ax)
ax.set_ylabel(format('Cumulative Count'))
ax.set_xlabel(format('Age'))
plt.legend(frameon=False, loc='upper left')
plt.tight_layout()
plt.show()
'''

data = extractData(['resident-year','EVD score','Ventriculostomy']).transpose()
data = data[data[:,2]=='1'] #Exclude Ommayas and VP shunts

excluded = ['','NA']
data = np.array([row for row in data if not any([item in excluded for item in row])]).astype(int)

y = [(data[(data[:,0] == year) * (data[:,1]==1)].sum(axis=0)[1],data[(data[:,0] == year)].sum(axis=0)[1]) for year in xrange(1,7)]

from scipy.stats import pearsonr

print pearsonr(range(1,7),[a/float(b) for a,b in y])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(np.arange(1,7),[a/float(b) for a,b in y], color=['k','k','k','gray','k','k'],alpha=0.7)
artist.adjust_spines(ax)
ax.set_xticks(np.arange(1,7)+0.4)
ax.set_xticklabels([r'\Large $\mathbf{%d} \; \left(\frac{%d}{%d}\right)$'%(i+1,a,b) for i,(a,b) in enumerate(y)])
ax.set_xlabel(r'\Large \textbf{Resident year}')
ax.set_ylabel(r'\Large \textbf{Accuracy} $\mathbf{\frac{1}{1+2+3}}$')
plt.tight_layout()
plt.show()

