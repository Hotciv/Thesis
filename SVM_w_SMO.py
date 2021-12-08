"""
    Author: Victor Barboza

    Exploring the implementation of Jon Charest at
    https://jonchar.net/notebooks/SVM/#Dual-form
"""

import numpy as np

# from gambit import LogitQRE

import matplotlib.pyplot as plt

# %matplotlib inline
# # This line is only needed if you have a HiDPI display
# %config InlineBackend.figure_format = 'retina'

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


class SMOModel:
    """Container object for the model used for sequential minimal optimization."""

    def __init__(
        self, X, y, C, alphas, b, errors, kernel_type="linear", tol=0.01, eps=0.01
    ):
        self.X = X  # training data matrix
        self.y = y  # class label vector (-1 or 1) for the data
        self.C = C  # regularization parameter
        self.kernels = {  # kernel types
            "linear": self.linear_kernel,
            "gaussian": self.gaussian_kernel,
        }
        self.kernel = self.kernels[kernel_type]  # kernel function
        self.alphas = alphas  # lagrange multiplier vector
        self.b = b  # scalar bias term
        self.errors = errors  # error cache
        self._obj = []  # record of objective function value
        self.m = len(self.X)  # store size of training set

        # Set tolerances
        self.tol = tol  # error tolerance
        self.eps = eps  # alpha tolerance

    def linear_kernel(self, x, y, b=1):
        """Returns the linear combination of arrays `x` and `y` with
        the optional bias term `b` (set to 1 by default)."""

        # Note the @ operator for matrix multiplication
        if b == 1:
            return x @ y.T + 1
        elif b == 0:
            return x @ y.T
        else:
            return x @ y.T + self.b

    def gaussian_kernel(self, x, y, sigma=1):
        """Returns the gaussian similarity of arrays `x` and `y` with
        kernel width parameter `sigma` (set to 1 by default)."""

        if np.ndim(x) == 1 and np.ndim(y) == 1:
            result = np.exp(-((np.linalg.norm(x - y, 2)) ** 2) / (2 * sigma ** 2))
        elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (
            np.ndim(x) == 1 and np.ndim(y) > 1
        ):
            result = np.exp(-(np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
        elif np.ndim(x) > 1 and np.ndim(y) > 1:
            result = np.exp(
                -(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2)
                / (2 * sigma ** 2)
            )
        return result

    # Objective function to optimize
    # TODO: understand why it is in reverse, according to the paper
    def objective_function(self, alphas):
        """Returns the SVM objective function based in the input model defined by:
        `alphas`: vector of Lagrange multipliers
        `target`: vector of class labels (-1 or 1) for training data
        `kernel`: kernel function
        `X_train`: training data for model."""

        return np.sum(alphas) - 0.5 * np.sum(
            (self.y[:, None] * self.y[None, :])
            * self.kernel(self.X, self.X)
            * (alphas[:, None] * alphas[None, :])
        )

    # Decision function (AKA constraint(s))
    def decision_function(self, x_test):
        """Applies the SVM decision function to the input feature vectors in `x_test`."""

        return (self.alphas * self.y) @ self.kernel(self.X, x_test) - self.b

    # TODO: probably going to change this or remove it
    def plot_decision_boundary(
        self, ax, xlabel='Feature 1', ylabel='Feature 2', resolution=100, colors=("b", "k", "r"), levels=(-1, 0, 1)
    ):
        """Plots the model's decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`)."""

        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate the model over the entire space
        xrange = np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), resolution)
        yrange = np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), resolution)
        grid = [
            [self.decision_function(np.array([xr, yr])) for xr in xrange]
            for yr in yrange
        ]
        grid = np.array(grid).reshape(len(xrange), len(yrange))

        # ax.margins(2, 2)

        # Plot decision contours using grid and
        # make a scatter plot of training data
        ax.contour(
            xrange,
            yrange,
            grid,
            levels=levels,
            linewidths=(1, 1, 1),
            linestyles=("--", "-", "--"),
            colors=colors,
        )
        ax.scatter(
            self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.winter, lw=0, alpha=0.25
        )

        # Plot support vectors (non-zero alphas)
        # as circled points (linewidth > 0)
        mask = np.round(self.alphas, decimals=2) != 0.0
        ax.scatter(
            self.X[mask, 0],
            self.X[mask, 1],
            c=self.y[mask],
            cmap=plt.cm.winter,
            lw=1,
            edgecolors="k",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.show()

        return grid, ax

    def take_step(self, i1, i2):

        # Skip if chosen alphas are the same
        if i1 == i2:
            return 0, self

        alph1 = self.alphas[i1]
        alph2 = self.alphas[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]
        s = y1 * y2

        # Compute L & H, the bounds on new possible alpha values
        if y1 != y2:
            L = max(0, alph2 - alph1)
            H = min(self.C, self.C + alph2 - alph1)
        elif y1 == y2:
            L = max(0, alph1 + alph2 - self.C)
            H = min(self.C, alph1 + alph2)
        if L == H:
            return 0, self

        # Compute kernel & 2nd derivative eta
        k11 = self.kernel(self.X[i1], self.X[i1])
        k12 = self.kernel(self.X[i1], self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])
        eta = 2 * k12 - k11 - k22
        # eta = k11 + k22 - 2 * k12

        # Compute new alpha 2 (a2) if eta is negative
        if eta < 0:
            a2 = alph2 - y2 * (E1 - E2) / eta
            # # Compute new alpha 2 (a2) if eta is positive
            # if (eta > 0):
            #     a2 = alph2 + y2 * (E1 - E2) / eta
            # Clip a2 based on bounds L & H
            if L < a2 < H:
                a2 = a2
            elif a2 <= L:
                a2 = L
            elif a2 >= H:
                a2 = H
            # if (a2 < L):
            #     a2 = L
            # elif (a2 > H):
            #     a2 = H

        # If eta is non-negative, move new a2 to bound with greater objective function value
        # # If eta is non-positive, move new a2 to bound with greater objective function value
        else:
            alphas_adj = self.alphas.copy()
            alphas_adj[i2] = L
            # objective function output with a2 = L
            Lobj = self.objective_function(alphas_adj)
            alphas_adj[i2] = H
            # objective function output with a2 = H
            Hobj = self.objective_function(alphas_adj)
            if Lobj > (Hobj + self.eps):
                a2 = L
            elif Lobj < (Hobj - self.eps):
                a2 = H
            # if Lobj < (Hobj - self.eps):
            #     a2 = L
            # elif Lobj > (Hobj + self.eps):
            #     a2 = H
            else:
                a2 = alph2

        # Push a2 to 0 or C if very close (threshold "function")
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (self.C - 1e-8):
            a2 = self.C

        # If examples can't be optimized within epsilon (eps), skip this pair
        if np.abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return 0, self

        # Calculate new alpha 1 (a1)
        a1 = alph1 + s * (alph2 - a2)

        # Update threshold b to reflect newly calculated alphas
        # Calculate both possible thresholds
        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self.b

        # Set new threshold based on if a1 or a2 is bound by L and/or H
        # TODO: understand this part?
        if 0 < a1 and a1 < self.C:
            b_new = b1
        elif 0 < a2 and a2 < self.C:
            b_new = b2
        # Average thresholds if both are bound
        else:
            b_new = (b1 + b2) * 0.5

        # Update model object with new alphas
        self.alphas[i1] = a1
        self.alphas[i2] = a2

        # Update error cache
        # TODO: where is the following from?
        # Error cache for optimized alphas is set to 0 if they're unbound
        for index, alph in zip([i1, i2], [a1, a2]):
            if 0.0 < alph < self.C:
                self.errors[index] = 0.0

        # Set non-optimized errors based on equation 12.11 in Platt's book
        # TODO: basically the same formula for computing threshold
        non_opt = [n for n in range(self.m) if (n != i1 and n != i2)]
        self.errors[non_opt] = (
            self.errors[non_opt]
            + y1 * (a1 - alph1) * self.kernel(self.X[i1], self.X[non_opt])
            + y2 * (a2 - alph2) * self.kernel(self.X[i2], self.X[non_opt])
            + self.b
            - b_new
        )

        # Update model threshold
        self.b = b_new

        return 1, self

    def examine_example(self, i2):

        y2 = self.y[i2]
        alph2 = self.alphas[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2

        # Proceed if error is within specified tolerance (tol)
        if (r2 < -self.tol and alph2 < self.C) or (r2 > self.tol and alph2 > 0):

            if len(self.alphas[(self.alphas != 0) & (self.alphas != self.C)]) > 1:
                # Use 2nd choice heuristic is choose max difference in error
                # TODO: understand this?
                if self.errors[i2] > 0:
                    i1 = np.argmin(self.errors)
                elif self.errors[i2] <= 0:
                    i1 = np.argmax(self.errors)
                step_result, self = self.take_step(i1, i2)
                if step_result:
                    return 1, self

            # Loop through non-zero and non-C alphas, starting at a random point
            for i1 in np.roll(
                np.where((self.alphas != 0) & (self.alphas != self.C))[0],
                np.random.choice(np.arange(self.m)),
            ):
                step_result, self = self.take_step(i1, i2)
                if step_result:
                    return 1, self

            # loop through all alphas, starting at a random point
            for i1 in np.roll(np.arange(self.m), np.random.choice(np.arange(self.m))):
                step_result, self = self.take_step(i1, i2)
                if step_result:
                    return 1, self

        return 0, self

    def train(self):

        numChanged = 0
        examineAll = 1

        while (numChanged > 0) or (examineAll):
            numChanged = 0
            if examineAll:
                # loop over all training examples
                for i in range(self.alphas.shape[0]):
                    examine_result, self = self.examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.objective_function(self.alphas)
                        self._obj.append(obj_result)
            else:
                # loop over examples where alphas are not already at their limits
                for i in np.where((self.alphas != 0) & (self.alphas != self.C))[0]:
                    examine_result, self = self.examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.objective_function(self.alphas)
                        self._obj.append(obj_result)
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1

        return self


# ################################################################
# ############## Using the linear kernel; example ################
# ################################################################

# X_train, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train, y)

# y[y == 0] = -1

# # Set model parameters and initial values
# C = 1000.0
# m = len(X_train_scaled)
# initial_alphas = np.zeros(m)
# initial_b = 0.0

# # Instantiate model
# model = SMOModel(X_train_scaled, y, C, initial_alphas, initial_b, np.zeros(m))

# # Initialize error cache
# initial_error = model.decision_function(model.X) - model.y
# model.errors = initial_error

# np.random.seed(0)
# model.train()

# fig, ax = plt.subplots()
# ax.margins(x=2, y=2)
# grid, ax = model.plot_decision_boundary(ax)

# print(np.count_nonzero(model.alphas.nonzero()))
# print(model.alphas.sum())
# # plt.plot(model.decision_function(model.X))
# # plt.show()

# # Add an outlier
# X_outlier = np.append(X_train_scaled, [0.1, 0.1])
# X_outlier = X_outlier.reshape(X_train.shape[0] + 1, X_train.shape[1])
# y_outlier = np.append(y, 1)

# # Set model parameters and initial values
# C = 1000.0
# m = len(X_outlier)
# initial_alphas = np.zeros(m)
# initial_b = 0.0

# # Instantiate model
# model = SMOModel(X_outlier, y_outlier, C, initial_alphas, initial_b, np.zeros(m))

# # Initialize error cache
# initial_error = model.decision_function(model.X) - model.y
# model.errors = initial_error

# model.train()
# fig, ax = plt.subplots()
# grid, ax = model.plot_decision_boundary(ax)

# # Set model parameters and initial values
# C = 1.0

# initial_alphas = np.zeros(m)
# initial_b = 0.0

# # Instantiate model
# model = SMOModel(X_outlier, y_outlier, C, initial_alphas, initial_b, np.zeros(m))

# # Initialize error cache
# initial_error = model.decision_function(model.X) - model.y
# model.errors = initial_error

# model.train()
# fig, ax = plt.subplots()
# grid, ax = model.plot_decision_boundary(ax)

# print(np.count_nonzero(model.alphas.nonzero()))
# print(np.array_equal(initial_alphas, model.alphas))

# ##################################################################
# ############## Using the Gaussian kernel; example ################
# ##################################################################
# X_train, y = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=1)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train, y)
# y[y == 0] = -1

# # Set model parameters and initial values
# C = 1.0
# m = len(X_train_scaled)
# initial_alphas = np.zeros(m)
# initial_b = 0.0

# # Instantiate model
# model = SMOModel(X_train_scaled, y, C, initial_alphas, initial_b, np.zeros(m), 'gaussian')

# # Initialize error cache
# initial_error = model.decision_function(model.X) - model.y
# model.errors = initial_error

# model.train()
# fig, ax = plt.subplots()
# grid, ax = model.plot_decision_boundary(ax)
