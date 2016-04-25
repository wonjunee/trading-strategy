import datetime as dt

def getNYSEdays(dt_start, dt_end, dt_timedelta):
	with open("NYSE_dates.txt") as F:
		days = [day.strip().split("/") for day in F]
		days = map(lambda x: dt.datetime(int(x[2]),int(x[0]),int(x[1])), days)
	return days
