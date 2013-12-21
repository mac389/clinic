import csv
import Graphics as artist
import utils as tech
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from pprint import pprint

import cPickle
import itertools

makeDict = lambda x: dict(zip(x,len(x)*['']))

rcParams['text.usetex']=True
filename = 'data.csv'
reader = csv.DictReader(open(filename))
data = list(reader)

device_types = ['evd','om','vps'] #Must filter out N/As
percentage = lambda x: (x['1']+x['2'])/float(sum(x.values()) if sum(x.values())>0 else 1)
categories = map(str,range(1,4))
shifts = ['L','R','0']
position = ['RF','LF','LO','RO']
pgy_years = map(str,range(1,7))

#Initialize dictionary
x = {}
for pos in position:
	for shift in shifts:
		x['%s-%s'%(pos,shift)] = makeDict(categories)

#Fill it
for pos in position:
	for shift in shifts:
		tmp = {category:len([datum for datum in data 
					if datum['midline shift direct'] == shift
					and datum['Postion'] == pos #sic
				  	and datum['Ventriculostomy Type'] == 'evd' 
				  	and datum['EVD score'] == category])
					for category in categories}
		x['%s-%s'%(pos,shift)] = (percentage(tmp),sum(tmp.values()))

stylize = lambda x: r'\Large \textbf{%s}'%s 
width = 0.2
colors = dict(zip(device_types,['k','r','b']))
fig,axs = plt.subplots(ncols=2,sharey=True)
for ax,pos in zip(axs,['LF','RF']):
	y = [x['%s-%s'%(pos,loc)][0] for loc in ['R','L','0']]
	inds = range(1,len(y)+1)
	ax.bar(inds,y, color='k')
	plt.hold(True)	
	artist.adjust_spines(ax)
	ax.set_ylabel(r'\Large \textbf{Fraction of EVDs placed accurately}')
	ax.set_xticks(inds)
	ax.set_xticklabels([r'\Large \textbf{%s, %s shift}'%(pos,loc) for loc in ['R','L','no']], rotation=45)
	ax.set_ylim(ymax=1.1)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()