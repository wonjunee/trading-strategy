def initialize(context):
    context.aapl = sid(24)
    context.spy = sid(8554)

# handle_data is ran for each minute.
# if can_trade(aapl) is true then take long position in the AAPL 60% of the portfolio value
# if can_trade(spy) is true then take short position in the SPY 40% of the portfolio value

def handle_data(context, data):
    # Note: data.can_trade() is explained in the next lesson
    if data.can_trade(context.aapl):
        order_target_percent(context.aapl, 0.60)
    if data.can_trade(context.spy):
        order_target_percent(context.spy, -0.40)
