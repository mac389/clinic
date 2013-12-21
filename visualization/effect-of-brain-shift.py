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
percentage = lambda x: (x['1']+x['2'])/float(sum(x.values()) if sum(x.values())>0 else 1)
categories = map(str,range(1,4))
shift = ['L','R','0']

x = {device_type:{} for device_type in device_types}
for device_type in device_types:
	x[device_type] = makeDict(shift)
	for e in shift:
		tmp = {category:len([datum for datum in data 
					if datum['midline shift direct'] == e
				  	and datum['Ventriculostomy Type'] == device_type 
				  	and datum['EVD score'] == category])
					for category in categories}
		x[device_type][e] = (percentage(tmp),sum(tmp.values()))

pprint(x)
#Effect of locale on accuracy of placement

width = 0.2
fig = plt.figure()
ax = fig.add_subplot(111)
colors = dict(zip(shift,['k','r','b']))
label = dict(zip(shift,['Left', 'Right','No']))
for i,loc in enumerate(shift):
	ax.bar(np.array(categories).astype(float) + i*width,[x[device_type][loc][0] for device_type in device_types],
		width,color=colors[loc], label=r'\textbf{%s Shift}'%label[loc])

artist.adjust_spines(ax)
ax.set_xlabel(r'\Large \textbf{Device}')
ax.set_ylabel(r'\Large \textbf{Fraction accurately placed}')
ax.set_xticks([1.2,2.2,3.2])
ax.set_xticklabels([r'\LARGE \textbf{%s}'%word.capitalize() for word in device_types])
ax.set_xlim((0.4,5))
ax.set_ylim(ymax=1.1)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

