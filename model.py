# model.py

import numpy as np
from kernels import kernel


class model(object):
    """
    Kernel model class for regression
    """

    def __init__(self, model_kernels):
        """
        Initialize information about the model
        """
        if not all(isinstance(k, kernel) for k in model_kernels):
            raise ValueError("model_kernels must all be instances of kernels.kernel")
        self._kernels = model_kernels
        self._data_dimensions = [k.data_dimensions() for k in self._kernels]
        self._bandwidth_dimensions = [k.bandwidth_dimensions() for k in self._kernels]
        self._kernel_data_indices = []
        self._kernel_bandwidth_indices = []
        self._kernel_index_by_bandwidth_index = []
        i = 0
        j = 0
        self._n_dimensions = 0
        self._n_bandwidths = 0
        for ki, (dd, bd) in enumerate(
            zip(self._data_dimensions, self._bandwidth_dimensions)
        ):
            self._kernel_data_indices.append(tuple(np.arange(i, i + dd)))
            self._kernel_bandwidth_indices.append(tuple(np.arange(j, j + bd)))
            self._kernel_index_by_bandwidth_index.extend([ki] * bd)
            self._n_dimensions += dd
            self._n_bandwidths += bd
            i += dd
            j += bd

    def evaluate(self, x, mu, bandwidth, index):
        """
        Evaluate the kernel function for regression
        index is the bandwidth index
        """
        j = self._kernel_index_by_bandwidth_index[index]
        return self._kernels[j].evaluate(x, mu, bandwidth)

    def evaluate_bandwidth_derivative(self, x, mu, bandwidth, index):
        """
        Evaluate the derivative of the kernel function with respect to bandwidth
        index is the bandwidth index
        """
        j = self._kernel_index_by_bandwidth_index[index]
        return self._kernels[j].evaluate_bandwidth_derivative(x, mu, bandwidth)
