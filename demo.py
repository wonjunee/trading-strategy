import random
import numpy as np
def main():
	age = random.uniform(20.0, 70.0)
	print age
	print np.random.normal(age,2.0)
if __name__ == "__main__":
	main()