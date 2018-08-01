"""
1. Covariance matrix is symmetric:
    the value of the elements in the covariance matrix is sigma_ij = E[(X_i - mu_i) (X_j - mu_j)^T] = sigma_ji
2. Further, the covariance matrix is positive-semidefinite
    w^T E[(X - EX) (X - EX)^T] w = E [w^T (X - EX)] [w^T (X - EX)]^T
                                         = E [w^T (X - EX)]^2
                                         >= 0
"""
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def plot_multivariate_normal(var_x, var_y, cov, mu_x=0, mu_y=0):
    mu_x, mu_y = mu_x, mu_y
    var_x, var_y = var_x, var_y
    cov = cov

    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X,Y = np.meshgrid(x,y)
    pos = np.array([X.flatten(),Y.flatten()]).T

    rv = multivariate_normal([mu_x, mu_y], [[var_x, cov], [cov, var_y]])

    fig = plt.figure(figsize=(10,10))
    ax0 = fig.add_subplot(111)
    ax0.contour(rv.pdf(pos).reshape(500,500))
    plt.show()

plot_multivariate_normal(15, 15, 10)