import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams
from matplotlib.mlab import psd
from scipy.stats import scoreatpercentile

#import api.utils as tech

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['text.usetex'] = True


format = lambda label: r'\Large \textbf{%s}'%str(label).capitalize()

def scatter(x,y,xlabel='',ylabel='', savename='test',show=False):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x,y)
	adjust_spines(ax)
	ax.set_xlabel(format(xlabel))
	ax.set_ylabel(format(ylabel))
	plt.savefig('%s.png'%savename,dpi=200)
	if show:
		plt.show()

def hist(data,xlabel,show=False,savename='test'):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(data)
	ax.axvline(x=np.median(data), linewidth=2,color='r', linestyle='--')
	adjust_spines(ax)
	ax.set_xlabel(format(xlabel.capitalize()))
	ax.set_ylabel(format('Count'))
	plt.savefig('%s.png'%savename, dpi=200)
	if show:
		plt.show()

def dashboard(data,numpc=3, labels = None,ed=False):
	if not ed:
		coeff,projections,latent = tech.princomp(data,numpc=numpc)
	else:
		coeff,projections,latent = data
	panels = {'projection':plt.subplot2grid((2,3),(0,0),colspan=2, rowspan=2),
			  'spectrum':plt.subplot2grid((2,3),(0,2)),
			  'silhouette':plt.subplot2grid((2,3),(1,2))}
	panels['projection'].scatter(projections[0],projections[1],s=30)
	adjust_spines(panels['projection'])
	panels['projection'].set_xlabel(r'\Large \textbf{Principal Component 1}')
	panels['projection'].set_ylabel(r'\Large \textbf{Principal Component 2}')

	cutoff=10
	panels['spectrum'].stem(range(1,cutoff+1),latent[:cutoff]/np.sum(latent))
	panels['spectrum'].set_xlim(0,cutoff+1)
	panels['spectrum'].set_ylim((0,1))
	adjust_spines(panels['spectrum'])
	panels['spectrum'].set_xlabel(r'\Large \textbf{Eigenvector}')
	panels['spectrum'].set_ylabel(r'\Large \textbf{Eigenvalue} $\left(\lambda\right)$')

	silhouettes = tech.silhouette(projections, k=8)
	idx = range(2,len(silhouettes)+2)
	panels['silhouette'].stem(idx,[silhouettes[x]['data'] for x in idx])

	#Get confidence intervals
	CIs = np.array([scoreatpercentile(silhouettes[x]['distribution'], 95) for x in idx])
	print CIs
	plt.hold(True)
	panels['silhouette'].plot(idx,CIs,linewidth=2,color='r',linestyle='--')
	adjust_spines(panels['silhouette'])
	panels['silhouette'].set_xlim((0,len(idx)+3))
	panels['silhouette'].set_ylim((0,1))
	panels['silhouette'].set_xlabel(r'\Large \textbf{Number of clusters}')
	panels['silhouette'].set_ylabel(r'\Large \textbf{Silhouette coefficient}')
	plt.tight_layout()

	rot = plt.figure()
	panel = rot.add_subplot(111)
	dt = coeff*(latent[:numpc]/np.sum(latent))
	dt_args = np.argsort(dt,axis=0)
	cax = panel.imshow(np.absolute(np.sort(coeff*(latent[:numpc]/np.sum(latent)),axis=0)[::-1]), aspect='auto',interpolation='nearest', vmin=-.15,vmax=0.15)
	adjust_spines(panel)
	if labels:
		panel.set_yticks(np.arange(len(labels)))
		panel.set_yticklabels(map(lambda word: word.capitalize(),labels))
	panel.set_xticks(np.arange(3))
	panel.set_xticklabels(np.arange(numpc)+1)
	panel.set_xlabel(r'\Large \textbf{Principal Component}')
	rot.colorbar(cax)
	plt.tight_layout()
	plt.grid(True)
	plt.show()

def heatmap(mat, ylabel='',xlabel='',xticklabels='', yticklabels='',
			pc_cutoff=None):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.imshow(mat,aspect='auto',interpolation='nearest')
	plt.colorbar(cax)

	if pc_cutoff:
		ax.axvline(pc_cutoff,color='r',linestyle='--', linewidth=2)

	adjust_spines(ax)
	ax.set_xlabel(format(xlabel))
	ax.set_ylabel(format(ylabel))

	if xticklabels != '':
		ax.set_xticks(range(len(xticklabels)))
		ax.set_xticklabels(map(format,xticklabels))
	

	if yticklabels != '':
		ax.set_yticks(range(len(yticklabels)))
		ax.set_yticklabels(map(format,yticklabels))
	
	plt.tight_layout()
	plt.show()

def adjust_spines(ax,spines=['bottom','left']):
	for loc, spine in ax.spines.iteritems():
		if loc in spines:
			spine.set_position(('outward',10))
			#spine.set_smart_bounds(True) #Doesn't work for log log plots
			spine.set_linewidth(1)
		else:
			spine.set_color('none') 
	if 'left' in spines:
		ax.yaxis.set_ticks_position('left')
	else:
		ax.yaxis.set_ticks([])

	if 'bottom' in spines:
		ax.xaxis.set_ticks_position('bottom')
	else:
		ax.xaxis.set_ticks([])

def scree_plot(eigVals,cutoff=0.95,savename=None, show=False,save=True,savebase=None):
	#Assume the list is all of the eigenvalues
	rel = np.cumsum(eigVals)/eigVals.sum()
	x = np.arange(len(rel))+1
	print eigVals.shape
	fig = plt.figure()
	ax = fig.add_subplot(111)
	line, = ax.plot(x,rel)
	line.set_clip_on(False)
	adjust_spines(ax,['bottom','left'])
	ax.set_xlabel(r'$\LARGE \lambda$')
	ax.set_ylabel('Fraction of variance')
	ax.set_xlim(0,len(eigVals))
	
	cutoff_idx = np.where(rel>cutoff)[0][0]
	
	ax.axvline(x=cutoff_idx, color='r',linestyle='--', linewidth=2)
	ax.axhline(y=rel[cutoff_idx],color='r',linestyle='--',linewidth=2)
	ax.tick_params(direction='in')
	ax.annotate(r" {\Large $\mathbf{\lambda=%d}$}" % cutoff_idx,xy=(.25, .9), xycoords='axes fraction', 
											horizontalalignment='center', verticalalignment='center')
	plt.tight_layout()
	if save:
		print savebase
		plt.savefig(savebase+'_scree.png',dpi=100)
			
	if show:
		plt.show()
	plt.close()

def fvt(traces,time):
	left,right = zip(*traces)
	panel_labels = ['Left','Right']
	fig, axs = plt.subplots(nrows=1,ncols=2, sharex=True, sharey=True)
	colors  = ['k','r','b','g']
	for j,(ax,data) in enumerate(zip(axs,[left,right])):
		for i,record in enumerate(data):
			ax.plot(time[::10],record[::10], colors[i],label=r'\Large \textbf{%d}'%(i+1))
			plt.hold(True)

		plt.legend(frameon=False, loc='upper left')
		ax.set_xlim(xmax=600)
		
		ax.annotate(r'\Large \textbf{%s}'%panel_labels[j], xy=(.1, .5),  xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center')
		adjust_spines(ax)
		ax.set_xlabel(r'\Large \textbf{Time (mins)}')
		ax.set_ylabel(r'\Large \textbf{Force (Arb. Units)}')
	plt.tight_layout()
	plt.show()

def ccf():	
	print 'Calculated'
	rowL=len(filenames)
	colL=rowL
	
	acf_panel,ax=subplots(rowL,colL, sharex=True, sharey=True) 
	#Should use absolute not relative normalization
	#Currently use absolute motivation
	for j in range(rowL):
		for i in range(colL):
			line, = ax[i,j].plot(arange(-w,w),ccfs[i+j], linewidth=2)
			line.set_clip_on(False)
			ax[i,j].axvline(x=0,color='r',linestyle='--', linewidth=2)
			postdoc.adjust_spines(ax[i,j],['bottom','left'])
			ax[i,j].spines['left'].set_smart_bounds(True)
			ax[i,j].spines['bottom'].set_smart_bounds(True)
			ax[i,j].set_ylabel('Covariance')
			ax[i,j].set_xlabel(r'Time $\left(ms\right)$')
			ax[i,j].set_axis_bgcolor('none')
			ax[i,j].tick_params(direction='in')
			ax[i,j].locator_params(nbins=(60/w))
			ax[i,j].annotate(r" {\Large $\mathbf{%s,%s}$}" %(tech.get_channel_id(filenames[i]),tech.get_channel_id(filenames[j])), 
							 xy=(.2, .8), xycoords='axes fraction',horizontalalignment='center', verticalalignment='center')
	tight_layout()
	savefig('test_ccf.png')