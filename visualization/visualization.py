import matplotlib.pyplot as plt
import numpy as np
import Graphics as artist
import api.utils as tech
import statsmodels.api as sm
from scipy.stats import scoreatpercentile

from matplotlib import rcParams
rcParams['text.usetex'] = True

formatted = {'tumor=1':'Tumor, Yes','tumor=0':'Tumor, No','tumor=':'Tumor, Unknown','total length (stripe length)':'Total Length',
	'tangent with vertical(angle)':'Tangent with vertical','skull thickness 2':'external hole thickness',
	'skull thickness 1':'inner hole thickness','sex=M':'male','sex=F':'female','sex=':'gender not known',
	'resident-year=NA':'Resident year: unknown','resident year=6':'PGY 6','resident year=5':"PGY 5",
	'resident year=5':'PGY 5','resident year=4':'PGY 4','resident year=3':'PGY 3','resident year=2':'PGY 2',
	'resident year=1':'PGY 1','midline shift direct=L':'Left shift','midline shift direct=R':'Right shift',
	'midline shift direct=0':'No midline shift','midline shift=':'Shift direction unknown',
	'length in brain (LP)':'Length in brain','burr hole diameter. Ext':'External burr hole diameter',
	'burr hole diameter 2. Int':'Internal burr hole diameter','Ventriculosomy=EVD':'Use of EVD',
	'Right Caudate to Septum Distance':'Distance from right caudate to septum',
	'Left Caudate to Septum Distance':'Distance from left caudate to septum',
	'Line through caudate heads to inner tables':'Distance from caudate to inner table','IVH?=1':'Intraventricular hemorrhage, Yes',
	'IVH?=0':'Intraventricular hemorrhage, No','IVH?=':'Intraventricular hemorrhage, Unknown'}
format = lambda text: r'\textbf{%s}'%(formatted[text].capitalize() if text in formatted else text.capitalize())

def covariance(heatmap,labels,show=False,savename=None,ml=False):
	#Covariance matrix
	fig = plt.figure(figsize=(13,13))
	ax = fig.add_subplot(111)
	cax = ax.imshow(heatmap,interpolation='nearest',aspect='equal')
	artist.adjust_spines(ax)
	ax.set_xticks(range(len(labels)))
	ax.set_xticklabels(map(artist.format,labels),range(len(labels)),rotation=90)

	ax.set_yticks(range(len(labels)))
	ax.set_yticklabels(map(artist.format,labels))

	if ml:
		ax.annotate(r'\LARGE \textbf{Training}', xy=(.2, .2),  xycoords='axes fraction',
		                horizontalalignment='center', verticalalignment='center')


		ax.annotate(r'\LARGE \textbf{Testing}', xy=(.7, .7),  xycoords='axes fraction',
		                horizontalalignment='center', verticalalignment='center')

	plt.colorbar(cax, fraction=0.10, shrink=0.8)
	plt.tight_layout()

	if savename:
		plt.savefig('%s.png'%savename,dpi=200)
	if show:
		plt.show()

def ecdf(data, show=False,savename=None):
	
 	ecdf = sm.distributions.ECDF(data)

 	x = np.linspace(data.min(),data.max())
 	y = ecdf(x)

 	cutoff = x[y>0.85][0]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x,y,'k',linewidth=3)
	artist.adjust_spines(ax)

	ax.annotate(r'\Large \textbf{Cutoff:} $%.03f$'%cutoff, xy=(.3, .2),  xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center')
	ax.set_xlabel(artist.format('Absolute Correlation'))
	ax.set_ylabel(artist.format('Percentile'))
	ax.axhline(y=0.85,color='r',linestyle='--',linewidth=2)
	ax.axvline(x=cutoff,color='r',linestyle='--',linewidth=2)
	ax.set_xlim((0,1))
	plt.tight_layout()

	if savename:
		plt.savefig('%s.png'%savename,dpi=200)
	if show:
		plt.show()

	return cutoff

def mvr_coefficients(model,labels,show=False,savename=None):

	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.imshow(model.coef_.transpose(),interpolation='nearest',aspect='auto',
		vmin=-0.5,vmax=0.5)
	artist.adjust_spines(ax)
	ax.set_yticks(range(len(labels)))
	ax.set_yticklabels(map(artist.format,[name for name in labels if 'EVD' not in name]))
	ax.set_xticks(range(3))
	ax.set_xticklabels(map(artist.format,range(1,4)))
	ax.set_xlabel(artist.format('Placement grade'))
	plt.colorbar(cax)
	plt.tight_layout()
	if savename:
		plt.savefig('%s.png'%savename,dpi=200)
	if show:
		plt.show()

def coefficients(model,labels,show=False,savename=None,title=None):

	fig = plt.figure(figsize=(8,10))
	ax = fig.add_subplot(111)
	x = -model.coef_.transpose()
	x /= np.absolute(x).max()
	y = np.arange(len(x))+0.5

	cutoff = scoreatpercentile(np.absolute(x),85)
	ax.barh(y,x,color=['r' if datum < 0 else 'g' for datum in x])
	ax.axvline(cutoff,linewidth=2,linestyle='--',color='r')
	ax.axvline(-cutoff,linewidth=2,linestyle='--',color='r')
	artist.adjust_spines(ax)
	ax.grid(True)
	ax.set_ylim(ymax=62)
	ax.set_xlim(xmin=-1.1,xmax=1.1)
	ax.set_yticks(y)
	ax.set_yticklabels(map(format,labels),y)
	ax.set_xlabel(format('Regression coefficient'))

	if title:
		ax.set_title(r'\Large \textbf{%s}'%title)
	plt.tight_layout()
	if show:
		plt.show()