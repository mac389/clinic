import csv, json

import numpy as np
#from visualization import Graphics as artist
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pprint import pprint

rcParams['text.usetex'] = True

TAB = '\t'
filename = 'covariance-matrix.tsv'

data = np.loadtxt('../data/%s'%filename, delimiter=TAB)
labels = open('temp-fields','rb').readlines()[:-2]
of_interest = data[:,2:5]

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(of_interest,interpolation='nearest', aspect='auto', vmin=-1,vmax=1)
#artist.adjust_spines(ax)
ax.set_yticks(range(len(labels)-2))
ax.set_xticks(range(3))
ax.set_yticklabels([r'\Large \textbf{%s}'%(x.rstrip('\n').capitalize()) for x in labels[:-3]],range(len(labels)-2))
ax.set_xticklabels([r'\Large \textbf{%s}'%x for x in ['1','2','3']])
ax.set_xlabel(r'\Large \textbf{Placement grade}')
cbar = plt.colorbar(cax)
plt.tight_layout()
plt.show()