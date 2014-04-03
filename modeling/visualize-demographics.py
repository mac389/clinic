import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import scoreatpercentile
from pprint import pprint
from matplotlib import rcParams

rcParams['text.usetex'] = True

format = lambda text: r'\Large \textbf{%s}'%text
data = np.loadtxt('../data/cov-matrix_.tsv',delimiter='\t')

with open('../data/demo-feature-names','rU') as f:
	labels = f.read().splitlines()


target = 'Volume of Brain Removed '
target_idx = labels.index(target)



roi =  0.5*data[target_idx,:]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(roi[~np.isnan(roi)],color='k',bins=100)
artist.adjust_spines(ax)
ax.set_xlabel(format('Correlation'))
ax.set_ylabel(format('Count'))
plt.tight_layout()
plt.show()