import csv
import json

import numpy as np
import utils as tech
import Graphics as artist


READ = 'rU'
WRITE = 'wb'
directory = json.load(open('directory.json',READ))

def convert(item):
	item = item.strip()
	if item == 'M' or item == 'vps' or item == 'ICU' or item == 'LF' or item == 'L':
		return -1
	elif item == 'F' or item == 'evd' or item == 'OR' or item == 'RF' or item == 'R':
		return 1
	elif item == 'om' or item == 'ED' or item == 'LT':
		return 0
	elif item == 'LO' or item == 'AG':
		return -2
	elif item == 'RO':
		return 2
	else:
		return item

cols = set(range(35))
bad_cols = set([0,1,2,12,13,29,30,10,11])
good_cols = list(cols-bad_cols)

with open('../Data/variables',READ) as f:
	vois = [x.rstrip('\t\n') for x in f.readlines()]

conversion = dict(zip(range(38),vois))
labels = [conversion[col] for col in good_cols if col != 10]

with open(directory['data'],READ) as fid:
	reader = csv.reader(fid)
	reader.next()

	data =np.array(filter(lambda row: '' not in row and 'NA' not in row and '?' not in row,
			[[convert(row[i]) for i in good_cols] for row in reader])).astype(float)


artist.dashboard(data, numpc=5,labels=labels)
