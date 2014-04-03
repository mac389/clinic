import numpy as np
import matplotlib.pyplot as plt
import Graphics as artist

from scipy.stats import scoreatpercentile
from pprint import pprint
from matplotlib import rcParams

rcParams['text.usetex'] = True

format = lambda text: r'\Large \textbf{%s}'%text
data = np.loadtxt('../data/cov-matrix_.tsv',delimiter='\t')

with open('../data/demo-feature-names','rU') as f:
	labels = f.read().splitlines()


target = 'Percentage of Tissue Removed 1 '
target_idx = labels.index(target)


exclusions =['Session ID','Date','Volume of Brain','Study ID','Session']

roi =  0.5*data[target_idx,:]
cutoff = scoreatpercentile(abs(roi),95)
significant_variables = [(labels[idx],roi[idx]) for idx in np.where(np.absolute(roi)>cutoff)[0] 
			if not any([exclusion in labels[idx] for exclusion in exclusions])]

pprint(significant_variables)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(roi[~np.isnan(roi)],color='k',bins=100, range=(-1,1))
artist.adjust_spines(ax)
ax.set_xlabel(format('Correlation'))
ax.set_ylabel(format('Count'))
ax.axvline(cutoff,linestyle='--',linewidth=2,color='r')
ax.axvline(-cutoff,linestyle='--',linewidth=2,color='r')
plt.tight_layout()
plt.show()
