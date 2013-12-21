import json
import csv

import numpy as np 
import matplotlib.pyplot as plt
import Graphics as artist

from sklearn.pls import PLSCanonical, PLSRegression, CCA
from pprint import pprint


READ = 'rU'
WRITE = 'wb'
directory = json.load(open('directory.json',READ))

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def convert(item):
	item = item.strip()
	if item == 'M' or item == 'vps' or item == 'ICU' or item == 'LF' or item == 'L':
		return -1
	elif item == 'F' or item=='evd' or item == 'OR' or item == 'RF' or item == 'R':
		return 1
	elif item == 'om' or item == 'ED' or item == 'LT':
		return 0
	elif item == 'LO' or item == 'AG':
		return -2
	elif item == 'RO':
		return 2
	else:
		return item

def broadcast(row):
	print row
	return row

#Could all this be done with a GNUplot script?
with open('../Data/variables',READ) as f:
	vois = [x.rstrip('\t\n') for x in f.readlines()]

translation = json.load(open('../Data/better-names.json',READ))

cols = set(range(35))
bad_cols = set([0,1,2,12,13,29,30])
good_cols = list(cols-bad_cols)

conversion = dict(zip(range(35),vois))
format = lambda cf: r'\textbf{%s}'%cf.capitalize()

labels = [translation[conversion[col]] for col in good_cols if col !=9]
with open(directory['data'],READ) as fid:
	reader = csv.reader(fid)
	reader.next()

	data =np.array(filter(lambda row: '' not in row and 'NA' not in row and '?' not in row,
			[[convert(row[i]) for i in good_cols] for row in reader if 'evd' in row])).astype(float)
	data = data[~np.isnan(data).any(axis=1)]
	loc = data[:,9]
	tumor = data[:,4]
	#data = (data- data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
	y = data[:,7]
	x = np.delete(data,7,1)

x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))

print x.shape

pls1 = PLSRegression(n_components = x.shape[1])
pls1.fit(x,y)


cfs = np.nan_to_num(np.log(pls1.coefs))

fig, (coeffs,dist) = plt.subplots(nrows=1,ncols=2)
coeffs.barh(range(len(cfs)),cfs, edgecolor='k',
		color = ['r' if x<0 else 'g' for x in cfs],
		linewidth=1)
artist.adjust_spines(coeffs)
coeffs.axvline(x=0,color='k',linestyle='--',linewidth=2)
coeffs.axvline(x=-5.3,color='r',linestyle='--',linewidth=1)
coeffs.axvline(x=5.3,color='r',linestyle='--',linewidth=1)
coeffs.set_xlabel(r'\Large \textbf{Importance} $\left(\log \beta\right)$')
coeffs.set_yticks(range(len(labels)))
coeffs.set_yticklabels(map(format,labels))
fig.tight_layout()

counts, bin_edges = np.histogram(abs(cfs[cfs!=0]), bins=10, normed=True)
cdf = np.cumsum(counts)
scale = 1.0/cdf[-1]
ncdf = scale * cdf
dist.plot(bin_edges[1:], ncdf,'k',linewidth=2)
dist.axhline(y=0.85,color='r',linestyle='--',linewidth=1)
dist.axvline(x=5.3,color='r',linestyle='--',
	linewidth=1)
artist.adjust_spines(dist)
dist.set_xlabel(r'\Large \textbf{Absolute Importance} $| \left(\log \beta \right)|$')
dist.set_ylabel(format('Percentile'))
plt.tight_layout()
plt.show()

