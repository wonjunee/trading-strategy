def initialize(context):
    # AAPL, MSFT, SPY
    context.security_list = [sid(24), sid(8554), sid(5061)]

# prints mean of securities over the last 10 minutes.
# data.history is pandas dataframe.
def handle_data(context, data):
    hist = data.history(context.security_list, 'volume', 10, '1m').mean()
    print hist.mean()