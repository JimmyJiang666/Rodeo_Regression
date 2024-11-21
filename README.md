# Rodeo_Regression

Implementation of the algorithm in *"Rodeo: Sparse, Greedy Nonparametric Regression"* by John Lafferty and Larry Wasserman.

## Features
- Adaptive bandwidth selection for local linear and constant regression.
- Dynamically prunes irrelevant dimensions for sparse, efficient regression.
- Supports Gaussian and Epanechnikov kernel types.
## Function Arguments Description

### **`rodeo_regression` Class**
#### `__init__(self, stat_model, data, responses, given_sigma2, weights=None, verbose=False, regression_type="linear")`
- **`stat_model`**: An instance of the `model` class containing kernel functions for each dimension.
- **`data`**: Training data (predictor variables) with shape `(n_features, n_samples)`, where `n_features` is the number of predictors and `n_samples` is the number of observations.
- **`responses`**: Training response variable with shape `(n_samples,)`.
- **`given_sigma2`**: Known variance of the response noise, used for variance estimation.
- **`weights`**: Optional weights for each data point, defaulting to uniform weights if not provided.
- **`verbose`**: If `True`, enables detailed logging of the optimization process.
- **`regression_type`**: Specifies the type of regression, either `"linear"` (local linear regression) or `"constant"` (local constant regression). Defaults to `"linear"`.

---

### **`local_rodeo` Method**
#### `local_rodeo(self, point, beta, h0=None)`
- **`point`**: Test point where the regression estimate is computed. A 1D array of length `n_features`.
- **`beta`**: Shrinkage factor for bandwidth adjustment. Must be in the range `(0, 1)`.
- **`h0`**: Initial bandwidths for each dimension. If not provided, they are automatically initialized based on the data.

**Returns**:
- **`h`**: Final bandwidths for the given test point.
- **`estimate`**: Regression estimate at the test point.

---

### **Private Helper Methods**
#### `_initialize_bandwidths(self, point, n_eff)`
- **`point`**: Test point for bandwidth initialization.
- **`n_eff`**: Effective sample size used for scaling the bandwidths.

**Returns**:
- **`h0`**: Initial bandwidths for each dimension.

#### `_construct_design_matrix(self, point, h)`
- **`point`**: Test point for constructing the design matrix.
- **`h`**: Current bandwidths for each dimension.

**Returns**:
- **`X`**: Design matrix used for regression.

#### `_compute_weights(self, point, h)`
- **`point`**: Test point for computing weights.
- **`h`**: Current bandwidths for each dimension.

**Returns**:
- **`W`**: Computed weights for all data points based on the kernel function.

#### `_fit_local_linear_regression(self, X, Y, W)`
- **`X`**: Design matrix.
- **`Y`**: Response vector.
- **`W`**: Weight vector for the observations.

**Returns**:
- **`S_x`**: Coefficient estimates (intercept and slopes for local linear regression).
- **`residuals`**: Residuals of the regression.

#### `_compute_L_j(self, point, h, j, kernel_type="gaussian")`
- **`point`**: Test point.
- **`h`**: Current bandwidths for all dimensions.
- **`j`**: Dimension index for which the derivative is computed.
- **`kernel_type`**: Type of kernel to use (`"gaussian"` or `"epanechnikov"`).

**Returns**:
- **`L_j`**: Derivative vector for the `j`-th dimension.

#### `_compute_Z_j(self, X, residuals, W, L_j)`
- **`X`**: Design matrix.
- **`residuals`**: Residuals from the regression.
- **`W`**: Weight vector.
- **`L_j`**: Derivative vector for the `j`-th dimension.

**Returns**:
- **`Z_j`**: Computed test statistic for bandwidth selection in the `j`-th dimension.

#### `_estimate_variance_Z_j(self, X, W, L_j)`
- **`X`**: Design matrix.
- **`W`**: Weight vector.
- **`L_j`**: Derivative vector for the `j`-th dimension.

**Returns**:
- **`s_j`**: Estimated standard deviation of the test statistic for the `j`-th dimension.
