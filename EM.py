"""
E-Step: estimate cluster responsibilities.
    A probability can be calculated for each of data point to each cluster.

M-Step: adjust parameters for specific distributions.
    In other words, M-step is about to maximize the likelihood of a distribution to data points. The way to
    do it is that, take these data points as samples from that distributions and calculate what parameters could
    have the distribution generate such data samples.
"""
from scipy.stats import multivariate_normal
import numpy as np

class EM(object):
    def __init__(self, data, centroids, init_cov=None, weights=None):
        self.data = data    # (num_data, feat_size)
        self.centroids = centroids  # (num_cent, feat_size)
        self.num_of_data = data.shape[0]
        self.feat_size   = data.shape[1]
        self.num_of_centroids = centroids.shape[0]

        if init_cov is None:
            pass
        else:
            self.init_cov = init_cov

        if weights is None:
            self.weights = np.array([1 / self.num_of_centroids] * self.num_of_centroids)
        else:
            self.weights = weights

    def compute_gaussian_pdf(self, x, mean, cov):
        return multivariate_normal.pdf(x, mean, cov)

    def compute_softmax(self, logits):
        softmax = logits.T / np.sum(logits, axis=1)
        return softmax.T


    def e_step(self):
        logits = np.zeros((self.num_of_data, self.num_of_centroids))

        # calculate pdf of all data points to each centroid
        for j in range(self.num_of_centroids):
            pdf = self.compute_gaussian_pdf(self.data, self.centroids[j], self.init_cov)
            logits[:, j] = self.weights[j] * pdf
        return self.compute_softmax(logits)

    def m_step(self, softmax):
        soft_count = np.sum(softmax, axis=0)    # num_centroids
        new_centroids = (self.data.T @ softmax) / soft_count
        new_centroids = new_centroids.T

        # TODO: replace with matrix multiplication
        new_cov = np.zeros((self.num_of_centroids, self.feat_size, self.feat_size))
        for j in range(self.num_of_centroids):
            diff = self.data - new_centroids[j]
            tmp_cov = np.zeros((self.feat_size, self.feat_size))
            for i in range(self.num_of_data):
                tmp_cov += softmax[i][j] * (diff[i].reshape(-1, 1) @ diff[i].reshape(1, -1))
            new_cov[j] = tmp_cov

        return new_centroids, new_cov


if __name__ == "__main__":
    data = [[10, 5],
            [2, 1],
            [3, 7]]

    centroids = [[3, 4],
                 [6, 3],
                 [4, 6]]

    em = EM(np.array(data),
            np.array(centroids),
            init_cov=np.array([[3, 0], [0, 3]]))

    softmax = em.e_step()
    centroids, cov = em.m_step(softmax)
    print("New centroids:", centroids)
    print("New cov:", cov)