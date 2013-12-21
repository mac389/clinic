import csv

import Graphics as artist
import matplotlib.pyplot as plt
import numpy as np
import utils as tech

from matplotlib import rcParams
from scipy.stats import scoreatpercentile

rcParams['text.usetex'] = True
rcParams['font.weight'] = 'bold'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

filename = 'demographics2.csv'
READ = 'rU'

#Control flow
histograms = False
pca = True

data = [row for row in csv.DictReader(open(filename,READ))]

CONSTANTS = [] 
useful_keys = [key for key in data[0].keys() if key.upper() not in CONSTANTS]

def dashboard(data,numpc=3):
	coeff,projections,latent = tech.princomp(data,numpc=numpc)
	panels = {'projection':plt.subplot2grid((2,3),(0,0),colspan=2, rowspan=2),
			  'spectrum':plt.subplot2grid((2,3),(0,2)),
			  'silhouette':plt.subplot2grid((2,3),(1,2))}
	panels['projection'].scatter(projections[0],projections[1],s=30)
	artist.adjust_spines(panels['projection'])
	panels['projection'].set_xlabel(r'\Large \textbf{Principal Component 1}')
	panels['projection'].set_ylabel(r'\Large \textbf{Principal Component 2}')

	cutoff=10
	panels['spectrum'].stem(range(1,cutoff+1),latent[:cutoff]/np.sum(latent))
	panels['spectrum'].set_xlim(0,cutoff+1)
	panels['spectrum'].set_ylim((0,1))
	artist.adjust_spines(panels['spectrum'])
	panels['spectrum'].set_xlabel(r'\Large \textbf{Eigenvector}')
	panels['spectrum'].set_ylabel(r'\Large \textbf{Eigenvalue} $\left(\lambda\right)$')

	silhouettes = tech.silhouette(projections, k=8)
	idx = range(2,len(silhouettes)+2)
	panels['silhouette'].stem(idx,[silhouettes[x]['data'] for x in idx])

	#Get confidence intervals
	CIs = np.array([scoreatpercentile(silhouettes[x]['distribution'], 95) for x in idx])
	plt.hold(True)
	panels['silhouette'].plot(idx,CIs,linewidth=2,color='r',linestyle='--')
	artist.adjust_spines(panels['silhouette'])
	panels['silhouette'].set_xlim((0,len(idx)+3))
	panels['silhouette'].set_ylim((0,1))
	panels['silhouette'].set_xlabel(r'\Large \textbf{Number of clusters}')
	panels['silhouette'].set_ylabel(r'\Large \textbf{Silhouette coefficient}')
	plt.tight_layout()

	rot = plt.figure()
	panel = rot.add_subplot(111)
	dt = coeff*(latent[:3]/np.sum(latent))
	dt_args = np.argsort(dt,axis=0)

	cax = panel.imshow(np.sort(coeff*(latent[:3]/np.sum(latent)),axis=0)[::-1], aspect='auto',interpolation='nearest', vmin=-.15,vmax=0.15)
	artist.adjust_spines(panel)
	
	panel.set_yticks(np.arange(len(useful_keys)))
	panel.set_yticklabels(map(lambda word: word.strip().capitalize(),[useful_keys[x[0]] for x in dt_args[::-1]]))
	panel.set_xticks(np.arange(3))
	panel.set_xticklabels(np.arange(3)+1)
	panel.set_xlabel(r'\Large \textbf{Principal Component}')
	rot.colorbar(cax)
	plt.tight_layout()
	plt.grid(True)
	plt.show()

# Look at variance in each key
if histograms:
	for key in sorted(useful_keys)[:2]:
		print key
		rec = [float(datum[key]) for datum in data 
				if datum[key] != 'NA' and datum[key] != '' 
				and datum[key] != '#DIV/0!']

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(rec, facecolor='k')

		artist.adjust_spines(ax)
		ax.axvline(np.median(rec), linestyle='--',color='r',linewidth=2)
		ax.set_ylabel(r'\Large \textbf{%s}'%key.upper())
		plt.tick_params(labelsize=20)

		locs, labels = plt.xticks()
		plt.xticks(locs, [r"$\Large \mathbf{%d }$" % x for x in locs]) 
		
		locs, labels = plt.yticks()
		plt.yticks(locs, [r"$\Large \mathbf{%d}$" % x for x in locs]) 

		plt.savefig('%s.png'%key,dpi=300)

if pca:

	data= np.genfromtxt('demographics2.csv', skip_header=1, missing_values = 'NA', 
				autostrip=True, filling_values = -1, delimiter=',')

	dashboard(data)
	
