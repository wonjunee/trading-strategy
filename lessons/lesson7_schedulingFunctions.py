# he following example takes a long position in SPY at the start of the week, 
# and closes out the position at 3:30pm on the last day of the week:

def initialize(context):
    context.spy = sid(8554)

    schedule_function(open_positions, date_rules.week_start(), time_rules.market_open())
    schedule_function(close_positions, date_rules.week_end(), time_rules.market_close(minutes=30))

def open_positions(context, data):
    order_target_percent(context.spy, 0.10)

def close_positions(context, data):
    order_target_percent(context.spy, 0)
