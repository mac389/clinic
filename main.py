import json,sys

from api.DataFrame import DataFrame


#Acquire and format data
filename = sys.argv[1] #Should make more robust with proper options parsing
dataframe = DataFrame(filename)

#Process data (visualize, find import features)

#Create model with data (use one group of important features to predict another group)

#Validate that model 

#Output results
