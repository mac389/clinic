import csv
import Graphics as artist
import utils as tech
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from pprint import pprint

import cPickle

makeDict = lambda x: dict(zip(x,len(x)*['']))

rcParams['text.usetex']=True
filename = 'data.csv'
reader = csv.DictReader(open(filename))
data = list(reader)

device_types = ['evd','om','vps'] #Must filter out N/As
percentage = lambda x: (x['1']+x['2'])/float(sum(x.values()))
categories = map(str,range(1,4))
edema = ['0','1']

x = {device_type:{} for device_type in device_types}
for device_type in device_types:
	x[device_type] = makeDict(edema)
	for e in edema:
		for category in categories:
			tmp = {category:len([datum for datum in data 
						if datum['mass effect'] == e
					  	and datum['Ventriculostomy Type'] == device_type 
					  	and datum['EVD score'] == category])
						for category in categories}
			x[device_type][e] = (percentage(tmp),sum(tmp.values()))

pprint(x)
#Effect of edema on accuracy of placement

width = 0.2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(map(int,categories),[x[device_type]['0'][0] for device_type in device_types],
		width,color='k', label=r'\textbf{No edema}')
plt.hold(True)
ax.bar(np.array(categories).astype(float)+width,[x[device_type]['1'][0] for device_type in device_types],
		width,color='r', label=r'\textbf{Edema}')
artist.adjust_spines(ax)
ax.set_xlabel(r'\Large \textbf{Device}')
ax.set_ylabel(r'\Large \textbf{Fraction placed accurately}')
ax.set_xticks([1.2,2.2,3.2])
ax.set_xticklabels([r'\LARGE \textbf{%s accuracy}'%word.capitalize() for word in device_types], rotation=45)
ax.set_xlim((0.4,5))
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
