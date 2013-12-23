import json,sys,cPickle,os

from api.DataFrame import DataFrame
from modeling.mvr import MVR
from modeling.Metanalysis import Metanalysis

from pprint import pprint
from time import time
from datetime import datetime
from progress.bar import Bar

#Acquire and format data
filename = sys.argv[1] #Should make more robust with proper options parsing
dataframe = DataFrame(filename)

timestamp = lambda time: datetime.fromtimestamp(time).strftime('%Y-%m-%d-%H-%M-%S')

iterations = 10
results = []
basedir = '../Data/%s/'%timestamp(time())
#Process data (visualize, find import features)

if not os.path.isdir(basedir):
	os.makedirs(basedir)

bar = Bar('Running cross-validation',max=iterations)
for iteration in xrange(iterations):
	model = MVR(dataframe,basedir=basedir,suffix=str(iteration))
	results.append(model.evaluation)
	del model
	bar.next()
bar.finish()

cPickle.dump(results,open('%sresults.evaluation'%basedir,'wb'))

#Model Selection 
selector = Metanalysis(results)
selector.display(savename='%saccuracy-distribution'%basedir)
