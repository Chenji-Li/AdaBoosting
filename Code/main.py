import numpy as np

from strongClassif import strongClassif
from Prediction import prediction


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ])
y = np.array([+1, +1, +1, -1, -1, +1, +1, -1, -1, -1])

AdaBoostingModel = strongClassif(x, y, 3)

#print(AdaBoostingModel)

#print(" The predictions of number 6 and 10 are:  ", prediction([6, 10], AdaBoostingModel))
