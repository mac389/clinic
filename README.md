clinic (0.5.0)
======

This repostitory contains tools to make the rigorous analysis of large clinical datases more accesible to the general community. A working knowledge of basic statistics and Python is assumed. By basic stats knowledge I mean the concepts of mean, median, and p-values. By basic Python I mean the ability to install a Python module and its dependencies on a local machine and type one-line Python commands in Terminal. 

###Installation

###Quickstart

###Pipeline
   						   Observations
						  HR   BP   RR  etc.
					     ----------------->
			   		    |
			Patient 	|
			 data		|                 ===> DataFrame ---> Model Development  <--> Model Validation   
					    |
					    V


`Clinic` takes __Patient data__, in the form of an `XLS`, `XLSX`, or `CSV` file, as its input. Each row represents a patient. Each column represents a type of data, such as heart rate, blood pressure, or respiratory rate. (See the Wiki for details on handling missing values and detecting data types.)

`Clinic` converts these data into a __DataFrame__, that contains the __Patient data__ and metadata that `Clinic` needs to process and store its analysis.

From __DataFrame__, `Clinic` develops a model to predict the outcome measure and validates that model. 