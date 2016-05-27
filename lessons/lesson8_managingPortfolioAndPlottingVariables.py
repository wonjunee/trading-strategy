def initialize(context):
    context.aapl = sid(24)
    context.spy = sid(8554)

    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open())
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())

def rebalance(context, data):
    order_target_percent(context.aapl, 0.50)
    order_target_percent(context.spy, -0.50)

def record_vars(context, data):

    long_count = 0
    short_count = 0

    # context.portfolio.positions is a dictionary that contains the current positions.
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            long_count += 1
        if position.amount < 0:
            short_count += 1

    # Plot the counts
    record(num_long=long_count, num_short=short_count)