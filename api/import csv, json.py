import csv, json

import numpy as np
from visualization import Graphics as artist

from pprint import pprint

TAB = '\t'
filename = 'covariance-matrix.tsv'

data = np.loadtxt('../data/%s'%filename, delimiter=TAB)

of_interest = data[:,2:5]

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(of_interest,interpolation='nearest', aspect='auto')
artist.adjust_spines(ax)
cbar = plt.colorbar(cax)
plt.tight_layout()
plt.show()