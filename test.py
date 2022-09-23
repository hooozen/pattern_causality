from pc import PC
import numpy as np

test_data = np.loadtxt('test_data/mutualism.csv', delimiter=',')
X = test_data[0]
Y = test_data[1]

pc = PC(X, Y)
pc.cacl()
