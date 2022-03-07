"""
    Author: Victor Barboza

    Assuming every slide of the window will update X, y, b, w, slack, 
    and maybe alphas and errors
"""
from matplotlib.pyplot import contour
from numpy.core.defchararray import count
from sklearn.cluster import KMeans
from SVM_w_SMO import *


class AAOSVM(SMOModel):
    """
        Reproduction of the Adversary-Aware Online SVM
    """

    def __init__(
        self,
        X,
        y,
        C,
        alphas,
        b,
        errors,
        m=100,
        Gp=250,
        Em=1,
        Er=1,
        Ym=1,
        Yr=1,
        s=0.6,
        kernel_type="linear",
        k=3,  # should be 2, 3, or 4...
    ):
        # Variables related to SVM with SMO
        super().__init__(X, y, C, alphas, b, errors, kernel_type)
        # if len(X) > 0:
        #     self.w = np.zeros((len(X[0]),))
        # else:
        self.w = np.zeros(1)  # weight vector

        # Variable(s) related to "relaxing" the SVM
        # self.slack = np.ones((len(X[0]),)) * s        # slack vector
        self.slack = s  # slack variable

        # Game theory variables
        self.mem = m  # memory parameter
        self.Gp = Gp  # period to approximate sequential equilibrium strategies
        self.Em = Em  # reward for classifying malicious correctly
        self.Er = Er  # reward for classifying regular correctly
        self.Ym = Ym  # punishment for misclassifying malicious
        self.Yr = Yr  # punishment for misclassifying regular

        # Helper variables
        self.Psi = 0  # last Psi value
        self.muM = 0  # probability distribution of the malicious websites
        self.muR = 0  # probability distribution of the regular websites
        self.clusters = None  # clusters of websites
        self.p = np.array([1 / k for i in range(k)])  # vector of probabilities of types
        self.k = k
        self.T = 2  # maximum number of iterations for each train() optimization

    def reset(self, X, y):
        # Set model parameters and initial values
        C = 100.0
        # C = 1000.0  # for SVM_w_SMO comparison
        # C = 1.0  # for SVM_w_SMO comparison
        # m = len(ds[1])  # for SVM_w_SMO comparison

        m = 2
        initial_alphas = np.zeros(m)
        initial_b = 0.0

        # Slidding window
        Sx = X[0]
        Sy = y[0]
        Sx = np.append([Sx], [X[1]], 0)
        Sy = np.append([Sy], [y[1]], 0)

        # Initialize model
        model = AAOSVM(Sx, Sy, C, initial_alphas, initial_b, np.zeros(m))

        # Initialize error cache
        model.errors = model.decision_function(model.X) - model.y

        model.w = np.zeros(len(X[0]))

        return model

    def get_clusters(self):
        """
        Gets clusters in X using an clustering algorithm
        """
        self.clusters = KMeans(n_clusters=self.k, random_state=42).fit(
            self.X
        )

        self.clusters.R = sum(self.y == -1)
        self.clusters.M = sum(self.y == 1)

        # number of instances in a cluster
        self.clusters.Z = np.zeros(self.k)
        for i in range(self.k):
            self.clusters.Z[i] = sum(self.clusters.labels_ == i)

        # for i, l in enumerate(self.clusters.labels_):
        #         # number of instances in a cluster that have label y
        #         if self.y[i] == y:
        #             count += 1
        #     # number of instances that have label 'M'
        #     if self.y[i] == 1:
        #         self.clusters.M += 1
        #     # number of instances that have label 'R'
        #     if self.y[i] == -1:
        #         self.clusters.R += 1
        # if not ((self.clusters.labels_ == 1) == (self.y == 1)).all():
        #     input('Yay!')

    def distance(self, a, b):
        """
        Euclidean distance between two np.arrays
        """
        return np.linalg.norm(a - b)

    def get_type(self, x):
        types = [self.distance(x, c) for c in self.clusters.cluster_centers_]
        return np.argmin(types)

    def utility(self, y, x, h, binary=True):
        """
        Classifier Utility
        Uc(y, x, C(x))
        """
        # Original formulas
        if binary:
            # t = R
            if y == -1:
                # incorrect classification
                if h == 1:
                    return -self.Yr * self.decision_function(np.ones(len(x)) - x)

                # correct classification
                else:
                    return self.Er * self.decision_function(np.ones(len(x)) - x)

        # Generalized formulas
        else:
            # t = R
            if y == -1:
                aux = np.where(x == 0, np.nan, np.where(x != 0, 0, x))
                aux = np.where(aux == np.nan, 1, aux)

                # incorrect classification
                if h == 1:
                    return -self.Yr * self.decision_function(aux)

                # correct classification
                else:
                    return self.Er * self.decision_function(aux)

        # t = M
        if y == 1:
            # incorrect classification
            if h == -1:
                return -self.Ym * self.decision_function(x)

            # correct classification
            else:
                return self.Em * self.decision_function(x)

    def update_probabilities(self, X):
        """
        Calculates the probability of a type
        The only part of game theory that uses the whole 
        """
        self.p = np.zeros(self.k)

        predicted = self.clusters.predict(X)
        for i in range(self.k):
            self.p[i] = sum(predicted == i)

        size_dataset = len(X)
        for i in range(self.k):
            self.p[i] /= size_dataset

    def phi(self, y, z):
        """
        Probability that an instance is from a cluster and has type y
        phi(x|(y, z))
        """
        count = 0
        # print(self.clusters.predict(np.reshape(x, (1, -1))))
        for i, l in enumerate(self.clusters.labels_):
            # number of instances in a cluster that have label y
            if l == z and self.y[i] == y:
                count += 1

        # print(count, y)
        # print(count/self.clusters.Z[z])
        return count / self.clusters.Z[z]

    def mu(self, y, z, x):
        """
        A belief, a probability distribution.
        u((y, z)|x)
        Represents the probability that given a message, we are dealing with a certain type of message
        """
        count_R = 0
        count_M = 0
        for i, c in enumerate(self.clusters.labels_):
            if c == z and self.y[i] == 1:
                count_M += 1
            elif c == z and self.y[i] == -1:
                count_R += 1
        # t = (M,xi)
        if y == 1:
            return (
                self.phi(1, z)
                * self.p[z]
                / (
                    count_R / (count_R + count_M)
                    + count_M / (count_R + count_M) * self.phi(1, z)
                )
            )

        # t = (R,xj)
        elif self.clusters.predict(np.reshape(x, (1, -1))) == z:
            return self.p[z] / (
                count_R / (count_R + count_M)
                + count_M / (count_R + count_M) * self.phi(1, z)
            )

        # t = (R,xj), j != i
        else:
            return 0

    def psi(self, x):
        """
        Helper function from Psi(x)
        """
        mu_M = sum([self.mu(1, i, x) for i in range(self.k)])
        mu_R = sum([self.mu(-1, i, x) for i in range(self.k)])
        aux = mu_M / mu_R
        return aux * (self.Em + self.Ym) / (self.Er + self.Yr)

    def update_psi(self, x):
        """
        Function that incorporates prior knowledge into the SVM
        """
        self.Psi = (1 + self.psi(x)) / (sum(self.w) + 2 * self.b)

    def update_weights(self, i1, i2, a1, a2):
        """
        Update the value of a couple of weights?
        """
        self.w = (
            self.w
            + self.y[i1] * (a1 - self.alphas[i1]) * self.X[i1]
            + self.y[i2] * (a2 - self.alphas[i2]) * self.X[i2]
        )

    # Objective function to optimize
    def objective_function(self, alphas):
        """Returns the SVM objective function based in the input model defined by:
        `alphas`: vector of Lagrange multipliers
        `target`: vector of class labels (-1 or 1) for training data
        `kernel`: kernel function
        `X_train`: training data for model."""

        return 0.5 * np.sum(
            (self.y[:, None] * self.y[None, :])
            * (self.X @ self.X.T)
            * (alphas[:, None] * alphas[None, :])
        ) - np.sum(alphas)

    # Decision function (AKA constraint(s))
    def decision_function(self, x_test):
        """Applies the SVM decision function to the input feature vectors in `x_test`."""
        # print('alphas', self.alphas)
        # print('y', self.y)
        # print('b', self.b)
        # print('kernel', self.X @ x_test.T)
        return (self.alphas * self.y) @ (self.X @ x_test.T) - self.b

    def predict(self, x_test: np.ndarray, show_loss=False):
        """
        Prediction function.

        Parameters:
            x_test (np.ndarray): instances to be predicted.
            show_loss (bool): option to return the possible loss.

        Returns:
            y_pred (np.array): predicted labels of x_test.
            loss (int): number of instances not exactly predicted.\
                (only if show_loss == True)
        """
        y_pred = self.decision_function(self, x_test)

        neg = y_pred == -1
        pos = y_pred == 1
        bin = neg | pos

        loss = len(y_pred[~bin])

        y_pred[y_pred < 0] = -1
        y_pred[y_pred >= 0] = 1
        
        if show_loss:
            return y_pred, loss
        else:
            return y_pred

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
        k11 = self.X[i1] @ self.X[i1].T
        k12 = self.X[i1] @ self.X[i2].T
        k22 = self.X[i2] @ self.X[i2].T
        # eta = 2 * k12 - k11 - k22
        eta = k11 + k22 - 2 * k12

        # # Compute new alpha 2 (a2) if eta is negative
        # if eta < 0:
        #     a2 = alph2 - y2 * (E1 - E2) / eta
        # Compute new alpha 2 (a2) if eta is positive
        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2) / eta

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

        # # If eta is non-negative, move new a2 to bound with greater objective function value
        # If eta is non-positive, move new a2 to bound with greater objective function value
        else:
            alphas_adj = self.alphas.copy()
            alphas_adj[i2] = L
            # objective function output with a2 = L
            Lobj = 0.5 * np.sum(
                (self.y[:, None] * self.y[None, :])
                * (self.X @ self.X.T)
                * (alphas_adj[:, None] * alphas_adj[None, :])
            ) - np.sum(alphas_adj)
            alphas_adj[i2] = H
            # objective function output with a2 = H
            Hobj = 0.5 * np.sum(
                (self.y[:, None] * self.y[None, :])
                * (self.X @ self.X.T)
                * (alphas_adj[:, None] * alphas_adj[None, :])
            ) - np.sum(alphas_adj)
            # if Lobj > (Hobj + self.eps):
            #     a2 = L
            # elif Lobj < (Hobj - self.eps):
            #     a2 = H
            if Lobj < (Hobj - self.eps):
                a2 = L
            elif Lobj > (Hobj + self.eps):
                a2 = H
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
        if 0 < a1 and a1 < self.C:
            b_new = b1
        elif 0 < a2 and a2 < self.C:
            b_new = b2

        # Average thresholds if both are bound
        else:
            b_new = (b1 + b2) * 0.5

        # Update model weight and alpha vectors
        self.update_weights(i1, i2, a1, a2)
        self.alphas[i1] = a1
        self.alphas[i2] = a2

        # Update error cache
        # Note: do not know where the following is from
        # Error cache for optimized alphas is set to 0 if they're unbound
        for index, alph in zip([i1, i2], [a1, a2]):
            if 0.0 < alph < self.C:
                self.errors[index] = 0.0

        # Set non-optimized errors based on equation 12.11 in Platt's book
        # Note: basically the same formula for computing threshold
        non_opt = [n for n in range(self.m) if (n != i1 and n != i2)]
        self.errors[non_opt] = (
            self.errors[non_opt]
            + y1 * (a1 - alph1) * (self.X[i1] @ self.X[non_opt].T)
            + y2 * (a2 - alph2) * (self.X[i2] @ self.X[non_opt].T)
            + self.b
            - b_new
        )

        # Update model threshold
        self.b = b_new

        return 1, self

    def train(self):

        numChanged = 0
        examineAll = 1
        t = 0

        # print('While')
        while t < self.T and ((numChanged > 0) or (examineAll)):
            numChanged = 0
            if examineAll:
                # print('For 1')
                # loop over all training examples
                for i in range(self.alphas.shape[0]):
                    examine_result, self = self.examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.objective_function(self.alphas)
                        self._obj.append(obj_result)
            else:
                # print('For 2')
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

            t += 1

        return self

    def partial_fit(self, X, Sx, Sy, x, y, i):

        # Update parameters before anything else
        self.X = Sx
        self.y = Sy

        # if the instance is considered phishing
        # if y * (self.w @ x + self.b) < self.slack:
        if y * (self.w @ x + self.b) * self.Psi < self.slack:

            # Initialize error cache  TODO: why is this here?
            self.errors = self.decision_function(self.X) - self.y

            prev_errors = self.errors.copy()

            # Train model
            self.train()

            # # Update scores
            # for j, err in enumerate(self.errors):
            #     # Decreased the error and Malicious sample
            #     if abs(err) < abs(prev_errors[j]) and Sy[j] == 1:
            #         self.Em -= self.Em * np.tanh(err)
            #     # Decreased the error and Regular sample
            #     elif abs(err) < abs(prev_errors[j]) and Sy[j] == -1:
            #         self.Er -= self.Er * np.tanh(err)
            #     # Increased the error and Malicious sample
            #     elif abs(err) > abs(prev_errors[j]) and Sy[j] == 1:
            #         self.Ym -= self.Ym * np.tanh(err)
            #     # Increased the error and Regular sample
            #     elif abs(err) > abs(prev_errors[j]) and Sy[j] == -1:
            #         self.Yr -= self.Yr * np.tanh(err)

        # Increasing the size of the parameters of the model
        if self.mem > self.m:
            self.m += 1
            self.alphas = np.append(self.alphas, [0], 0)
            self.errors = np.append(self.errors, [0], 0)
            # print(len(Sx), len(Sy), len(self.alphas), len(self.errors))

        # if reached the limit of the window
        # i.e. window is full and now will move
        if Sy.shape[0] >= self.mem:
            Sx = Sx[1:]
            Sy = Sy[1:]

        if i % self.Gp == 0:
            # k = 3  # should be 2, 3, or 4...
            # self.update_probabilities(k)
            # print(self.p)
            # input()
            self.get_clusters()
            self.update_probabilities(X)
            # self.phi(-1, 0)
            # self.phi(-1, 1)
            # self.phi(1, 1)
            # self.phi(-1, 2)
            # print(self.psi(x))
            # print(self.Psi)
            self.update_psi(x)
            # print(self.Psi)
            # print(self.p)
            # input()

        # Adding instance to the window
        Sx = np.append(Sx, [x], 0)
        Sy = np.append(Sy, [y], 0)

        return Sx, Sy

