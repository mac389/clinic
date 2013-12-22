import json,sys,cPickle

from api.DataFrame import DataFrame
from modeling.mvr import MVR
from pprint import pprint
from time import time
from datetime import datetime
#Acquire and format data
filename = sys.argv[1] #Should make more robust with proper options parsing
dataframe = DataFrame(filename)


timestamp = lambda time: datetime.fromtimestamp(time).strftime('%Y-%m-%d-%H-%M-%S')

iterations = 10000
results = []
#Process data (visualize, find import features)
for iteration in xrange(iterations):
	print iteration,'\t',
	model = MVR(dataframe,dir=timestamp(time()),suffix=str(iteration))
	results.append(model.evaluation)
	del model

cPickle.dump(results,open('results.evaluation','wb'))