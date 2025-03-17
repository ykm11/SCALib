import numpy as np
import math

class OnlineCorrVector:

    def __init__(self, dim):
        self._n = 0
        self._dim = dim
        self._mean_x = np.zeros(dim, dtype=np.float64)
        self._M2_x = np.zeros(dim, dtype=np.float64)
        self._mean_y = 0.0
        self._M2_y = 0.0
        self._Cov = np.zeros(dim, dtype=np.float64)


    def update(self, x, y):
        """
        x: numpy.array, shape (dim,)
        y: scaler
        """

        self._n += 1
        if self._n == 1:
            self._mean_x = x.astype(np.float64)
            self._mean_y = y
        else:
            delta_x = x - self._mean_x
            delta_y = y - self._mean_y
            # Update mean
            self._mean_x += delta_x / self._n
            self._mean_y += delta_y / self._n
            # Update (co)variance
            self._Cov += delta_x * (y - self._mean_y)
            self._M2_x += delta_x * (x - self._mean_x)
            self._M2_y += delta_y * (y - self._mean_y)


    def clear(self):
        self._n = 0
        self._mean_x[:] = 0
        self._M2_x[:] = 0
        self._mean_y = 0.0
        self._M2_y = 0.0
        self._Cov[:] = 0


    def covariance(self):
        """Get covariance (element-wise)"""
        if self._n < 2:
            return np.full(self._dim, np.nan)
        return self._Cov / (self._n - 1)


    def pearson(self):
        """
        Calculating the Pearson correlation coefficient
        """
        if self._n < 2:
            return np.full(self._dim, np.nan)
        var_x = self._M2_x / (self._n - 1)
        var_y = self._M2_y / (self._n - 1)
        # If the variance of Y is zero, it returns nan
        if var_y == 0:
            return np.full(self._dim, np.nan)
        # Elements where their variance are zero will be nan
        r = np.where(var_x == 0, np.nan,
                self.covariance() / np.sqrt(var_x * var_y))
        return r
