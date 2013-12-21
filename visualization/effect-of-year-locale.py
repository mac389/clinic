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
filename = '../Data/11-13-2013.csv'
reader = csv.DictReader(open(filename,'rU'))
data = list(reader)

device_types = ['evd','om','vps'] #Must filter out N/As
percentage = lambda x: (x['1']+x['2'])/float(sum(x.values()) if sum(x.values())>0 else 1)
categories = map(str,range(1,4))
locale = ['ICU','OR']
pgy_years = map(str,range(1,7))

x = {device_type:{} for device_type in device_types}
for device_type in device_types:
	x[device_type] = makeDict(locale)
	for e in locale:
		x[device_type][e] = makeDict(pgy_years)
		for year in pgy_years:
			tmp = {category:len([datum for datum in data 
					if datum['Location'] == e
				  	and datum['Ventriculostomy'] == device_type 
				  	and datum['EVD score'] == category
				  	and datum['resident-year'] == year])
					for category in categories}
			x[device_type][e][year] = (percentage(tmp),sum(tmp.values()))

stylize = lambda x: r'\Large \textbf{%s}'%s 
width = 0.2

pprint(x)
colors = dict(zip(device_types,['k','r','b']))
fig,axs = plt.subplots(ncols=2, sharey=True)
for (loc,ax) in zip(locale[::-1],axs):
	for i,device in enumerate(device_types):
		ax.bar(np.array(pgy_years).astype(float)+i*width,
			[x[device][loc][year][0] for year in pgy_years],width,
			label = r'\Large \textbf{%s}'%device, 
			color=colors[device])

	artist.adjust_spines(ax)
	ax.set_xlabel(r'\Large \textbf{Postgraduate year}')
	ax.set_ylabel(r'\Large \textbf{Fraction accurately placed in {\LARGE %s}}'%loc.upper())
	ax.set_ylim(ymax=1.1)
	ax.set_xlim((0,7))
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
