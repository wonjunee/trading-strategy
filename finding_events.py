import pandas as pd
import numpy as np
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep
import matplotlib.pyplot as plt




def load_symlists(sym_list):
	lines = open(sym_list)
	ls_symbols = [i.strip() for i in lines]
	return ls_symbols

def main():
	# This demo is for simulating the strategy

	# Variables
	dt_start = dt.datetime(2015,3,30)
	dt_end = dt.datetime(2016,3,30)

	sym_list = 'sp5002012.txt'
	market_sym = 'SPY'

	starting_cash = 100000
	bol_period = 20


	print "Setting Up ..."
	# Obtatining data from Yahoo
	ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))
	dataobj = da.DataAccess('Yahoo')
	ls_symbols = load_symlists(sym_list)
	ls_symbols.append(market_sym)

	# key values. Creating a dictionary.
	ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
	ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
	d_data = dict(zip(ls_keys, ldf_data))

	# fill out N/A values
	for s_key in ls_keys:
	    d_data[s_key] = d_data[s_key].fillna(method='ffill')
	    d_data[s_key] = d_data[s_key].fillna(method='bfill')
	    d_data[s_key] = d_data[s_key].fillna(1.0)

	# df_close contains only a close column.
	df_close = d_data['close']
	df_volume = d_data['volume']

	print "Finding Events ..."
	''' 
	Finding the event dataframe 
	'''
	ts_market = df_close['SPY']

	# Creating an empty dataframe
	df_events = copy.deepcopy(df_close)
	df_events = df_events * np.NAN

	# Time stamps for the event range
	ldt_timestamps = df_close.index

	print "Searching for the events of interest ..."

	rolling_mean = pd.rolling_mean(df_close,window=bol_period)
	rolling_std = pd.rolling_std(df_close,window=bol_period)

	rolling_mean_vol = pd.rolling_mean(df_volume,window=bol_period)
	rolling_std_vol = pd.rolling_std(df_volume,window=bol_period)


	target_sym = 'SPY'


	for num_ind in xrange(11):
		a = 0.0
		for delay in range(1,11):

			num = float(num_ind)/10+1
			

			vline = [] # mark days of interest
			pline = [] # a day after vline

			for i in range(len(rolling_mean.index)):
				time = rolling_mean.index[i]
				if df_close.loc[time,target_sym] > rolling_mean.loc[time,target_sym] - num*rolling_std.loc[time,target_sym] \
					and df_volume.loc[time,target_sym] > rolling_mean_vol.loc[time,target_sym] + num*rolling_std_vol.loc[time,target_sym]:
					vline.append(time)
					try:
						pline.append(rolling_mean.index[i+delay])
					except:
						pline.append(rolling_mean.index[i])

			# caclulate percentage of predictability
			vline_price = np.array(map(lambda x: df_close.loc[x,target_sym], vline))
			pline_price = np.array(map(lambda x: df_close.loc[x,target_sym], pline))

			print "num: %.2f" %(num), "delay:",delay,"num_events:",len(vline_price), "average percent: %.2f" %(100*np.mean((pline_price - vline_price)/vline_price)), "%"
			a += np.mean((pline_price - vline_price)/vline_price)

			if delay == 10:
				print "average: %.2f" %(100.0*a/5.0), "%"

	# Subplots
	
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].plot(ldt_timestamps, df_close[target_sym], 'b-', \
			 rolling_mean.index, rolling_mean[target_sym]+num*rolling_std[target_sym], 'g--', \
			 rolling_mean.index, rolling_mean[target_sym]-num*rolling_std[target_sym], 'r--')
	
	axarr[1].plot(ldt_timestamps, df_volume[target_sym], 'b-', \
			 rolling_mean.index, rolling_mean_vol[target_sym]+num*rolling_std_vol[target_sym], 'g--', \
			 rolling_mean.index, rolling_mean_vol[target_sym]-num*rolling_std_vol[target_sym], 'r--')
	# plt.axvline(x=ldt_timestamps[10])
	for i in vline:
		axarr[0].axvline(x=i)
		axarr[1].axvline(x=i)
	plt.show()

	bol_val = (df_close-rolling_mean)/rolling_std

if __name__ == "__main__":
	main()