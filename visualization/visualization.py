import matplotlib.pyplot as plt
import numpy as np
import Graphics as artist
import api.utils as tech
import statsmodels.api as sm


from matplotlib import rcParams
rcParams['text.usetex'] = True

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
