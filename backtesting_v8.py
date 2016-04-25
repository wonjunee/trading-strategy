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
	"""
	This demo is for simulating the strategy
	Variables
	"""
	dt_start = dt.datetime(2013,1,1)
	dt_end = dt.datetime(2015,12,31)

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
	"""
	key values. Creating a dictionary.
	"""
	ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
	ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
	d_data = dict(zip(ls_keys, ldf_data))
	"""
	fill out N/A values
	"""
	for s_key in ls_keys:
	    d_data[s_key] = d_data[s_key].fillna(method='ffill')
	    d_data[s_key] = d_data[s_key].fillna(method='bfill')
	    d_data[s_key] = d_data[s_key].fillna(1.0)
	"""
	df_close contains only a close column.
	"""
	df_close = d_data['close']
	df_volume = d_data['volume']

	print "Finding Events ..."
	''' 
	Finding the event dataframe 
	'''
	ts_market = df_close['SPY']

	# Creating an empty dataframe
	df_events = copy.deepcopy(df_close) * 0

	# Time stamps for the event range
	ldt_timestamps = df_close.index

	rolling_mean = pd.rolling_mean(df_close,window=bol_period)
	rolling_std = pd.rolling_std(df_close,window=bol_period)

	rolling_mean_vol = pd.rolling_mean(df_volume,window=bol_period)
	rolling_std_vol = pd.rolling_std(df_volume,window=bol_period)

	'''
	finding_events starts here
	'''

	bol_clo = (df_close-rolling_mean)/rolling_std


	delays = 14

	for s_sym in ls_symbols:
		for i in range(1,len(ldt_timestamps)-delays):
			bol_tod = bol_clo[s_sym].loc[ldt_timestamps[i]]
			bol_yes = bol_clo[s_sym].loc[ldt_timestamps[i-1]]
			bol_tod_mark = bol_clo["SPY"].loc[ldt_timestamps[i]]


			if (bol_tod >= 3.0 and bol_yes <= 3.0 and bol_tod_mark <= -1.0):
				for delay in range(delays):
					df_events[s_sym].loc[ldt_timestamps[i+delay]] -= (30000.00/df_close[s_sym].loc[ldt_timestamps[i]])
					if df_close[s_sym].loc[ldt_timestamps[i+delay]] < df_close[s_sym].loc[ldt_timestamps[i]]:
						break
			elif (bol_tod <= -3.0 and bol_yes >= -3.0 and bol_tod_mark >= 1.0):
				for delay in range(delays):
					df_events[s_sym].loc[ldt_timestamps[i+delay]] += (30000.00/df_close[s_sym].loc[ldt_timestamps[i]])
					if df_close[s_sym].loc[ldt_timestamps[i+delay]] > df_close[s_sym].loc[ldt_timestamps[i]]:
						break
			elif (bol_tod <= -2.0 and bol_yes >= -2.0 and bol_tod_mark >= 1.0):
				for delay in range(delays):
					df_events[s_sym].loc[ldt_timestamps[i+delay]] -= (10000.00/df_close[s_sym].loc[ldt_timestamps[i]])
					if df_close[s_sym].loc[ldt_timestamps[i+delay]] < df_close[s_sym].loc[ldt_timestamps[i]]:
						break
			elif (bol_tod >= 2.0 and bol_yes <= 2.0 and bol_tod_mark <= -1.0):
				for delay in range(delays):
					df_events[s_sym].loc[ldt_timestamps[i+delay]] += (10000.00/df_close[s_sym].loc[ldt_timestamps[i]])
					if df_close[s_sym].loc[ldt_timestamps[i+delay]] > df_close[s_sym].loc[ldt_timestamps[i]]:
						break
			


	print "Starting Simulation ..."

	# Find symbols that satisfy the event condition.
	ls_symbols_red = []

	for sym in ls_symbols:
		for i in range(len(ldt_timestamps)):
			if df_events[sym].loc[ldt_timestamps[i]] != 0:
				ls_symbols_red.append(sym)
				break

	'''
	value and cash are zero arrays
	'''
	# df_orders = copy.deepcopy(df_events)
	print "ls_symbols_red", ls_symbols_red
	df_orders = df_events[ls_symbols_red]
	value = copy.deepcopy(df_events) * 0
	cash = copy.deepcopy(value[market_sym])

	'''
	Update value
	'''
	print "Updating Value and Cash Array..."
	for s_sym in ls_symbols_red:
	    for i in range(len(ldt_timestamps)):
	        ind_time = ldt_timestamps[i]
	        if i == 0:
	            if df_orders[s_sym].loc[ind_time] != 0:
	                sym_value = df_orders[s_sym].loc[ind_time] * df_close[s_sym].loc[ind_time] 
	                value[s_sym].loc[ind_time] = sym_value
	                cash[ind_time] -= sym_value
	        else:
	            ind_time_yest = ldt_timestamps[i-1]
	            if df_orders[s_sym].loc[ind_time] != 0 and df_orders[s_sym].loc[ind_time_yest] == 0:
	                sym_value = df_orders[s_sym].loc[ind_time] * df_close[s_sym].loc[ind_time] 
	                value[s_sym].loc[ind_time] = sym_value
	                cash[ind_time] -= sym_value
	            elif df_orders[s_sym].loc[ind_time_yest] != 0:
	                sym_value = df_orders[s_sym].loc[ind_time] * df_close[s_sym].loc[ind_time] 
	                value[s_sym].loc[ind_time] = sym_value
	                cash[ind_time] -= (df_orders[s_sym].loc[ind_time] - df_orders[s_sym].loc[ind_time_yest]) * df_close[s_sym].loc[ind_time_yest]
	'''
	Update cash
	'''
	cash.to_csv("c:/cash_pre.csv",sep=",",mode="w")
	print "Modifying Cash Array..."
	cash[ldt_timestamps[0]] += starting_cash
	for i in range(1,len(ldt_timestamps)):
	    ind_prev = cash[ldt_timestamps[i-1]]
	    ind_curr = cash[ldt_timestamps[i]]
	    cash[ldt_timestamps[i]] = ind_curr + ind_prev

	# Save to csv files
	cash.to_csv("c:/cash.csv",sep=",",mode="w")
	value.to_csv("c:/portfolio.csv",sep=",",mode="w")

	print "Updating Total..."
	for i in range(len(ldt_timestamps)):
	    sym_sum = 0
	    for s_sym in ls_symbols_red:
	        sym_sum += value[s_sym].ix[ldt_timestamps[i]]
	    cash[ldt_timestamps[i]] += sym_sum

	# Save to csv files
	cash.to_csv("c:/total.csv",sep=",",mode="w")
	ts_market.to_csv("c:/ts_market.csv",sep=",",mode="w")

	# Normalizing dataframes.
	cash /= cash[0]
	ts_market /= ts_market[0]

	print "Summary..."
	tot_ret_fund = cash[-1]
	tot_ret_mark = ts_market[-1] 

	'''
	Create new array for fund and market
	'''
	daily_ret_fund = np.zeros((len(ldt_timestamps),1))
	daily_ret_mark = copy.deepcopy(daily_ret_fund)

	for i in range(1,len(ldt_timestamps)):
	    daily_ret_fund[i] = cash[ldt_timestamps[i]]/cash[ldt_timestamps[i-1]]-1
	    daily_ret_mark[i] = ts_market[ldt_timestamps[i]]/ts_market[ldt_timestamps[i-1]]-1

	vol_fund = np.std(daily_ret_fund)
	vol_mark = np.std(daily_ret_mark)

	avg_ret_fund = np.average(daily_ret_fund)
	avg_ret_mark = np.average(daily_ret_mark)

	sharpe_fund = np.sqrt(252) * avg_ret_fund / vol_fund
	sharpe_mark = np.sqrt(252) * avg_ret_mark / vol_mark

	print "Start Date:", dt_start
	print "End Date  :", dt_end
	print " "
	print "Sharpe Ratio of Fund: ", sharpe_fund
	print "Sharpe Ratio of $SPX: ", sharpe_mark
	print " "
	print "Total Return of Fund: ", tot_ret_fund
	print "Total Return of $SPX: ", tot_ret_mark
	print " "
	print "Standard Deviation of Fund: ", vol_fund
	print "Standard Deviation of $SPX: ", vol_mark
	print " "
	print "Average Daily Return of Fund: ", avg_ret_fund
	print "Average Daily Return of $SPX: ", avg_ret_mark

	# plt.plot(cash.index, cash, 'r', ts_market.index, ts_market, 'b')
	# f, axarr = plt.subplots(3, sharex=True)
	# axarr[0].plot(cash.index, cash, 'r', ts_market.index, ts_market, 'b')
	# axarr[0].set_title('Testing')
	# axarr[1].plot(ts_market.index, df_volume["SPY"], 'b')
	# axarr[2].plot(ts_market.index, rolling_std["SPY"], 'b')
	# plt.show()


	# df_volume_norm = df_volume["SPY"]/df_volume["SPY"][ldt_timestamps[0]]

	f, axarr = plt.subplots(3, sharex=True)
	axarr[0].plot(cash.index, cash, 'r', ts_market.index, ts_market, 'b')
	axarr[0].set_title('Testing')
	axarr[1].plot(ts_market.index, df_volume["SPY"], 'b', ts_market.index, rolling_mean_vol["SPY"]+rolling_std_vol["SPY"], 'b--', ts_market.index, rolling_mean_vol["SPY"]-rolling_std_vol["SPY"], 'b--')
	axarr[2].plot(ts_market.index, rolling_std["SPY"], 'g')
	plt.show()

if __name__ == "__main__":
	main()