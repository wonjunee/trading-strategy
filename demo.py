import random
import numpy as np
import datetime as dt

def get_data(ldt_timestamps, ls_symbols, ls_keys):
	path = "c:/Python27/Lib/site-packages/QSTK/QSData/Yahoo/ACE.csv"
	with open(path,"r") as F:
		print [f.strip().split(',') for f in F]

def main():
	age = random.uniform(20.0, 70.0)
	print age
	print np.random.normal(age,2.0)

	print dt.timedelta(hours=16)
	get_data(1,2,3)
if __name__ == "__main__":
	main()