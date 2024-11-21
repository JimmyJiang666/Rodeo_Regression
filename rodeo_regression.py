# rodeo_regression.py

import numpy as np
from model import model


class rodeo_regression:
    """
    Regression Rodeo Algorithm for Bandwidth Selection in Local Linear Regression
    """

    def __init__(self, stat_model, data, responses, given_sigma2, weights=None, verbose=False, regression_type="linear"):
        self._stat_model = stat_model  # Instance of the model class
        self._data = data  # Predictor variables, shape: (n_features, n_samples)
        self._responses = responses  # Response variable, shape: (n_samples,)
        self._weights = weights if weights is not None else np.ones(data.shape[1])
        self.sigma2 = given_sigma2
        self.verbose = verbose
        self.regression_type = regression_type  # "linear" or "constant"

    def local_rodeo(self, point, beta, h0=None):
        assert 0 < beta < 1, "Beta must be between 0 and 1"
        n = self._data.shape[1]
        d = self._stat_model._n_bandwidths

        # Effective sample size accounting for weights
        v1 = np.sum(self._weights)
        v2 = np.sum(self._weights ** 2)
        n_eff = v1 ** 2 / v2
        w_eff = v2 / v1

        # Initialize bandwidths
        if h0 is None:
            # Estimate initial bandwidths based on data spread
            h0 = self._initialize_bandwidths(point, n_eff)
        h = h0.copy()

        # Activate all dimensions
        A = list(range(d))
        if self.verbose:
            print("Initial bandwidths:", h)
            print("n_eff:", n_eff)
        # Main loop
        while len(A) > 0:
            # Compute weights W
            W = self._compute_weights(point, h)

            # Construct design matrix X
            X = self._construct_design_matrix(point, h)

            # Fit local linear regression
            S_x, residuals = self._fit_local_linear_regression(X, self._responses, W)

            for j in A.copy():
                # Compute derivative L_j
                L_j = self._compute_L_j(point, h, j)

                # Compute Z_j
                Z_j = self._compute_Z_j(X, residuals, W, L_j)

                # Estimate variance s_j^2
                s_j = self._estimate_variance_Z_j(X, W, L_j)

                # Compute threshold lambda_j
                lambda_j = s_j * np.sqrt(2 * np.log(n_eff))
                if self.verbose:
                    print(f"Dimension {j}: Z_j = {Z_j}, lambda_j = {lambda_j}")
                # Update bandwidths
                if np.abs(Z_j) > lambda_j:
                    h[j] *= beta
                    if self.verbose:
                        print(f"Updating bandwidth {j}:", h[j])
                else:
                    if self.verbose:
                        print("Removing dimension:", j)
                    A.remove(j)

        # Final estimation at point
        W = self._compute_weights(point, h)
        X = self._construct_design_matrix(point, h)
        S_x, _ = self._fit_local_linear_regression(X, self._responses, W)
        if self.regression_type == "linear":
            estimate = S_x[0]  # Use the intercept term as the estimate
        elif self.regression_type == "constant":
            estimate = S_x

        return h, estimate

    def _initialize_bandwidths(self, point, n_eff):
        # Use data spread to estimate initial bandwidths
        d = self._stat_model._n_bandwidths
        h0 = np.zeros(d)
        for j in range(d):
            data_j = self._data[j, :]
            h0[j] = np.std(data_j - point[j]) + 1e-6  # Avoid zero
        # Scale initial bandwidths
        h0 *= np.log(np.log(n_eff) + 2)
        return h0

    def _construct_design_matrix(self, point, h):
        """
        Constructs the design matrix X for local regression.
        """
        n = self._data.shape[1]
        d = self._stat_model._n_bandwidths

        if self.regression_type == "constant":
            # Local constant regression: only intercept term
            X = np.ones((n, 1))
        elif self.regression_type == "linear":
            # Local linear regression: intercept + covariate terms
            X = np.ones((n, d + 1))  # First column is ones for the intercept
            for j in range(d):
                X[:, j + 1] = (self._data[j, :] - point[j]) / h[j]
        else:
            raise ValueError("Unsupported regression type. Choose 'linear' or 'constant'.")

        return X

    def _compute_weights(self, point, h):
        # Compute the weights W based on the kernel function
        n = self._data.shape[1]
        W = np.ones(n)
        for j in range(self._stat_model._n_bandwidths):
            K_values = self._stat_model.evaluate(
                self._data[j, :], point[j], h[j], j
            )
            W *= K_values
        W *= self._weights
        return W

    def _fit_local_linear_regression(self, X, Y, W):
        # Fit local linear regression and compute residuals
        WX = W[:, np.newaxis] * X  # Element-wise multiplication
        XTWX = X.T @ WX
        XTWY = X.T @ (W * Y)
        theta = np.linalg.solve(XTWX, XTWY)
        residuals = Y - X @ theta
        if self.regression_type == "constant":
            S_x = theta[0]  # In local constant regression, only intercept matters
        elif self.regression_type == "linear":
            S_x = theta  # Full coefficient vector
        return S_x, residuals

    # def _compute_L_j(self, point, h, j):
    #     # Compute the derivative vector L_j
    #     X_j = self._data[j, :]
    #     K_j = self._stat_model.evaluate(X_j, point[j], h[j], j)
    #     dK_j = self._stat_model.evaluate_bandwidth_derivative(
    #         X_j, point[j], h[j], j
    #     )
    #     L_j_diag = dK_j / K_j
    #     return L_j_diag

    def _compute_L_j(self, point, h, j, kernel_type="gaussian"):
        X_j = self._data[j, :]  # Data points for the j-th dimension
        h_j = h[j]
        u_j = (X_j - point[j]) / h_j  # Standardized coordinates in the j-th dimension

        if kernel_type == "gaussian":
            # L_j for Gaussian kernel: diag((X_ij - x_j)^2 / h_j^3)
            L_j = (u_j ** 2) / h_j ** 3  # Vectorized form, no diag needed in this setup

        elif kernel_type == "epanechnikov":
            # L_j for Epanechnikov kernel:
            # diag((2 * (X_ij - x_j)^2) / (h_j^3 * (5 - (X_ij - x_j)^2 / h_j^2)) * I(|X_ij - x_j| <= sqrt(5) * h_j))
            squared_diff = u_j ** 2
            indicator = (squared_diff <= 5)  # Boolean mask for |u_j| <= sqrt(5)
            denominator = (5 - squared_diff)
            # Avoid division by zero
            denominator[~indicator] = np.inf
            L_j = np.zeros_like(u_j)
            L_j[indicator] = (2 * squared_diff[indicator]) / (h_j ** 3 * denominator[indicator])

        else:
            raise ValueError("Unsupported kernel type. Choose 'gaussian' or 'epanechnikov'.")

        return L_j


    def _compute_Z_j(self, X, residuals, W, L_j):
        # Compute Z_j
        WX = W[:, np.newaxis] * X  # Shape: (n_samples, n_features + 1)
        XTWX = X.T @ WX  # Shape: (n_features + 1, n_features + 1)
        inv_XTWX = np.linalg.inv(XTWX)  # Inverse matrix

        # Compute B = (X^T W X)^{-1} X^T W
        B = inv_XTWX @ (X.T * W)  # Shape: (n_features + 1, n_samples)

        # Compute G_j
        G_j = B[0, :] * L_j  # Shape: (n_samples,)

        # Compute Z_j
        Z_j = np.sum(G_j * residuals)
        return Z_j


    def _estimate_variance_Z_j(self, X, W, L_j):
        # Assume sigma2 is known for now
        if self.sigma2:
            sigma2 = self.sigma2
        else:
            sigma2 = 0.3 ** 2
        if self.verbose:
            print("sigma2:", sigma2)
        # Reuse computation of B
        WX = W[:, np.newaxis] * X
        XTWX = X.T @ WX
        inv_XTWX = np.linalg.inv(XTWX)
        B = inv_XTWX @ (X.T * W)

        # Compute G_j
        G_j = B[0, :] * L_j  # Shape: (n_samples,)

        # Compute s_j^2
        s_j2 = sigma2 * np.sum(G_j ** 2)
        s_j = np.sqrt(s_j2)
        return s_j

