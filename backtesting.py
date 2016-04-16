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

	rolling_mean = pd.rolling_mean(df_close,window=bol_period)
	rolling_std = pd.rolling_std(df_close,window=bol_period)

	rolling_mean_vol = pd.rolling_mean(df_volume,window=bol_period)
	rolling_std_vol = pd.rolling_std(df_volume,window=bol_period)

	'''
	finding_events starts here
	'''

	bol_clo = (df_close-rolling_mean)/rolling_std
	bol_vol = (df_volume-rolling_mean_vol)/rolling_std_vol

	for s_sym in ls_symbols:
	    for i in range(1,len(ldt_timestamps)-5):
	    	bol_clo_tod = bol_clo[s_sym].loc[ldt_timestamps[i]]
	    	bol_vol_tod = bol_vol[s_sym].loc[ldt_timestamps[i]]

	    	if bol_clo_tod < 1.5 and bol_vol_tod >1.5:
	    		df_events[s_sym].loc[ldt_timestamps[i]] = 1

	# create an order array
	print "Creating An Order Array ..."
	# orders = np.array([[dt.datetime(2008,1,1),'1','1',1]])
	orders = np.zeros((1,4))
	# tmp = copy.deepcopy(orders)

	for i in range(len(ldt_timestamps)):
	    for s_sym in ls_symbols:
	        if df_events[s_sym].ix[ldt_timestamps[i]]==1:
	            # Buy Order
	            # tmp.iloc[0,0] = ldt_timestamps[i] # date
	            # tmp.iloc[0,1] = s_sym # symbol
	            # tmp.iloc[0,2] = "Buy"
	            # tmp.iloc[0,3] = 100
	            tmp = [[ldt_timestamps[i], s_sym, "Buy", 100]]
	            
	            orders = np.append(orders,tmp,axis=0)
	         
	            # Sell order 5 days later
	            # if i+5<len(ldt_timestamps):
	            #     tmp[0,0] = ldt_timestamps[i+5]
	            #     tmp[0,1] = s_sym
	            #     tmp[0,2] = "Sell"
	            #     tmp[0,3] = 100
	            #     orders = np.append(orders,tmp,axis=0)
	            # else:
	            #     tmp[0,0] = ldt_timestamps[-1]
	            #     tmp[0,1] = s_sym
	            #     tmp[0,2] = "Sell"
	            #     tmp[0,3] = 100
	            #     orders = np.append(orders,tmp,axis=0)

	# Deleting the first temp row 
	orders = np.delete(orders,0,0)

	# Sort orders by date
	orders = orders[orders[:,0].argsort()]

	print orders
	
	# Start simulation
	print "Starting Simulation ..."
	
	ls_symbols = []
	for i in orders:
	    if i[1] in ls_symbols:
	        pass
	    else:
	        ls_symbols.append(i[1])
	order_symbols = copy.deepcopy(ls_symbols)
	
	# Construct date list
	dt_start = orders[0,0]
	dt_end = orders[-1,0]

	ls_symbols, df_close, d_data, ldt_timestamps = setup_reduced(ls_symbols, dt_start, dt_end, market_sym)

	ts_market = df_close[market_sym]

	# Construct event data frame
	df_events = copy.deepcopy(df_close)

	# df_events is zero array
	df_events *= 0

	# value and cash are zero arrays
	value = copy.deepcopy(df_events)
	cash = copy.deepcopy(value[market_sym])

	# Find from orders and insert into df_events
	for i in range(len(orders)):
	    if orders[i,2] == 'Buy':
	        df_events[orders[i,1]].ix[orders[i,0]] += orders[i,3]
	    else:
	        df_events[orders[i,1]].ix[orders[i,0]] -= orders[i,3]

	# Fill out zeros in df_events
	for s_sym in ls_symbols:
	    for i in range(1,len(ldt_timestamps)):
	        ind_prev = df_events[s_sym].ix[ldt_timestamps[i-1]]
	        ind_curr = df_events[s_sym].ix[ldt_timestamps[i]]

	        df_events[s_sym].ix[ldt_timestamps[i]] = ind_curr + ind_prev

	# Update value
	print "Updating Value and Cash Array..."
	for s_sym in ls_symbols:
	    for i in range(len(ldt_timestamps)):
	        ind_time = ldt_timestamps[i]
	        if i == 0:
	            if df_events[s_sym].ix[ind_time] > 0:
	                sym_value = df_events[s_sym].ix[ind_time] * df_close[s_sym].ix[ind_time] 
	                value[s_sym].ix[ind_time] = sym_value
	                cash[ind_time] -= sym_value
	        else:
	            ind_time_yest = ldt_timestamps[i-1]
	            if df_events[s_sym].ix[ind_time] > 0 and df_events[s_sym].ix[ind_time_yest]==0:
	                sym_value = df_events[s_sym].ix[ind_time] * df_close[s_sym].ix[ind_time] 
	                value[s_sym].ix[ind_time] = sym_value
	                cash[ind_time] -= sym_value
	            elif df_events[s_sym].ix[ind_time_yest]>0:
	                sym_value = df_events[s_sym].ix[ind_time] * df_close[s_sym].ix[ind_time] 
	                value[s_sym].ix[ind_time] = sym_value
	                cash[ind_time] -= (df_events[s_sym].ix[ind_time] - df_events[s_sym].ix[ind_time_yest]) * df_close[s_sym].ix[ind_time]

	# Update cash
	print "Modifying Cash Array..."
	cash[ldt_timestamps[0]] += starting_cash
	for i in range(1,len(ldt_timestamps)):
	    ind_prev = cash[ldt_timestamps[i-1]]
	    ind_curr = cash[ldt_timestamps[i]]

	    cash[ldt_timestamps[i]] = ind_curr + ind_prev

	print "Updating Total..."
	for i in range(len(ldt_timestamps)):
	    sym_sum = 0
	    for s_sym in ls_symbols:
	        sym_sum += value[s_sym].ix[ldt_timestamps[i]]
	    cash[ldt_timestamps[i]] += sym_sum

	cash_raw = copy.deepcopy(cash)

	cash /= cash[0]
	ts_market /= ts_market[0]

	print "Calculating Total Return..."
	tot_ret_fund = cash[-1]
	tot_ret_mark = ts_market[-1] 

	print "Calculating Volatility..."

	# Create new array for fund and market
	daily_ret_fund = np.zeros((len(ldt_timestamps),1))
	daily_ret_mark = copy.deepcopy(daily_ret_fund)

	for i in range(1,len(ldt_timestamps)):
	    daily_ret_fund[i] = cash[ldt_timestamps[i]]/cash[ldt_timestamps[i-1]]-1
	    daily_ret_mark[i] = ts_market[ldt_timestamps[i]]/ts_market[ldt_timestamps[i-1]]-1

	vol_fund = np.std(daily_ret_fund)
	vol_mark = np.std(daily_ret_mark)

	print "Calculating Average Daily Return..."
	avg_ret_fund = np.average(daily_ret_fund)
	avg_ret_mark = np.average(daily_ret_mark)

	print "Calculating Sharpe Ratio..."
	sharpe_fund = np.sqrt(252) * avg_ret_fund / vol_fund
	sharpe_mark = np.sqrt(252) * avg_ret_mark / vol_mark

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


if __name__ == "__main__":
	main()