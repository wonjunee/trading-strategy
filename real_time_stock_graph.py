import requests
import json
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
	sym = "XOM"
	price = []
	plt.ion()
	plt.show()
	plt.axis([0,300, 84, 85])		
	path = "http://download.finance.yahoo.com/d/quotes.csv?s={0}&e=.csv&f=sl1v".format(sym)
	while True:
		real_time = requests.get(path).text
		line = real_time.split(",")
		print line
		price.append(json.loads(line[1]))
		plt.plot(price)
		plt.draw()
		time.sleep(1)

if __name__=='__main__':
	main()
