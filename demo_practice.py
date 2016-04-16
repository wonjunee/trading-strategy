import pandas as pd
import numpy as np

index = range(10)
columns = ["spy","xom","tsla"]

data = np.zeros((10,3))

for i in range(10):
	for j in range(3):
		data[i,j] = i+j
print data


df = pd.DataFrame(data,index=index, columns=columns)

df.to_csv("c:/asdf.csv",sep=',',mode="w")