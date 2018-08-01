import numpy as np
from scipy.stats import multivariate_normal


x = np.array([[1,2],
              [4,5],
              [7,8],
              [10,12]])

centroid = np.array([1,2])

a = multivariate_normal.pdf(x, mean=centroid, cov=[[3, 0], [0, 3]])
print(a)
print(type(a))
