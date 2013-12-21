import csv
import json
import argparse

from pprint import pprint

READ = 'rU'
WRITE = 'wb'

directory = json.load(open('directory.json',READ))
db = csv.DictReader(open(directory['data'],READ))

#Could all this be done with a GNUplot script?
parser = argparse.ArgumentParser(description='Analyze ventriculostomy data')
parser.add_argument('vars')
args = parser.parse_args()
print args.echo