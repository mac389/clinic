import numpy as np
import matplotlib.pyplot as plt

from visualization import Graphics as artist
from matplotlib import rcParams

rcParams['text.usetex'] = True
 
class Metanalysis(object):

	def __init__(self,models):
		self.models = models #models is a list of (contingency table, accuracy) tuples

		#select the best model 
		#change how I define best later

		_,self.accuracy = zip(*self.models)

		self.best_idx = np.argmax(self.accuracy)
		print self.best_idx

	def display(self, show=False,savename=None):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(self.accuracy,color='k')
		artist.adjust_spines(ax)
		ax.set_xlabel(artist.format('Accuracy'))
		ax.set_ylabel(artist.format('Number of models'))

		plt.tight_layout()
		if savename:
			plt.savefig('%s.png'%savename,dpi=200)
		if show:
			plt.show()