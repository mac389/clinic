import csv
import Graphics as artist
import utils as tech
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from pprint import pprint

rcParams['text.usetex']=True
filename = 'data.csv'
reader = csv.DictReader(open(filename))
data = list(reader)


#Figure 1 Ventriculostomy placement scores by year
pgy_years = map(str,range(1,7))
placement_categories =  map(str,range(1,4))
x = {}
percentage = lambda x: (x['1']+x['2'])/float(sum(x.values()))
for pgy_year in pgy_years:
	tmp = dict(zip(placement_categories,[len([datum for datum in data 
					if datum['resident PGY'] == pgy_year
				  	and datum['EVD score'] == category]) 
				for category in placement_categories]))
	x[pgy_year] = (percentage(tmp),sum(tmp.values()))


fig = plt.figure()
ax = fig.add_subplot(111)
width=0.6
for pgy_year in pgy_years:
	ax.bar(int(pgy_year),[x[pgy_year][0]], width, color='k', 
			alpha = 1 if x[pgy_year][1] > 10 else 0.2)
artist.adjust_spines(ax)
ax.set_xlabel(r'\Large \textbf{Postgraduate year}')
ax.set_ylabel(r'\Large \textbf{Fraction accurately placed}')
ax.set_title(r'\LARGE \textbf{Ventriculostomy Accuracy}')
stylize = lambda x: r'\Large \textbf{%s}'%x
def formatter(x,pos):
	return stylize(x)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter))
ax.set_xticks(np.array(pgy_years).astype(float)+0.5*width)
ax.set_ylim(ymax=1.1)
ax.set_xticklabels([r'\Large \textbf{%s}, $\left(n=%d\right)$'%(pgy_year,x[pgy_year][1]) for pgy_year in pgy_years])
plt.show()