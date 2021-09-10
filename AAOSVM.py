"""
    Author: Victor Barboza

    Assuming every slide of the window will update X, y, b, w, slack, 
    and maybe alphas and errors
"""
from sklearn.cluster import KMeans
from SVM_w_SMO import *
from time import time


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

    def get_clusters(self, k):
        '''
        Gets clusters in X using an clustering algorithm
        '''
        self.clusters = KMeans(n_clusters=k, random_state=0).fit(self.X)  # TODO: remove random state
        if not ((self.clusters.labels_ == 1) == (self.y == 1)).all():
            input('Yay!')

    
    def calculate_p(self, cluster):
        '''
        Calculates the probability of a type
        '''
        pass
    
    def mu(self, t, x):
        '''
        A belief, a probability distribution.
        Represents the probability that given a message, we are dealing with a certain type of message
        '''
        pass

    def psi(self, x):
        """
        Helper function from Psi(x)
        """
        aux = 1  # TODO: change to sum of mus
        return aux*(self.Em + self.Ym)/(self.Er + self.Yr)

    def Psi(self, x):
        """
        Function that incorporates prior knowledge into the SVM
        """
        self.Psi = (1 + psi(x)) / (self.w.T * np.ones((len(self.X[0]),)) + 2 * self.b)

    def update_weights(self, i1, i2, a1, a2):
        """
        Update the value of a couple of weights?
        """
        # TODO: change the vector multiplication to instances?
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

        return np.sum(alphas) - 0.5 * np.sum(
            (self.y[:, None] * self.y[None, :])
            * self.kernel(self.X, self.X)
            * (alphas[:, None] * alphas[None, :])
        )

    # Decision function (AKA constraint(s))
    def decision_function(self, x_test):
        """Applies the SVM decision function to the input feature vectors in `x_test`."""

        return (self.alphas * self.y) @ self.kernel(self.X, x_test) - self.b

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

        # Update model weight and alpha vectors
        self.update_weights(i1, i2, a1, a2)
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

    # def train(self):

    #     numChanged = 1
    #     examineAll = 0

    #     examine_result, self = self.examine_example(-1)
    #     if examine_result:
    #         obj_result = self.objective_function(self.alphas)
    #         self._obj.append(obj_result)

    #     while(numChanged > 0) or (examineAll):
    #     # while(numChanged > 0):  # or (examineAll):
    #         numChanged = 0
    #         if examineAll:
    #             # loop over all training examples
    #             for i in range(self.alphas.shape[0]):
    #                 examine_result, self = self.examine_example(i)
    #                 numChanged += examine_result
    #                 if examine_result:
    #                     obj_result = self.objective_function(self.alphas)
    #                     self._obj.append(obj_result)
    #             print('ExamineAll')
    #         else:
    #             # loop over examples where alphas are not already at their limits
    #             for i in np.where((self.alphas != 0) & (self.alphas != self.C))[0]:
    #                 examine_result, self = self.examine_example(i)
    #                 numChanged += examine_result
    #                 if examine_result:
    #                     obj_result = self.objective_function(self.alphas)
    #                     self._obj.append(obj_result)
    #                 # print('Alphas')
    #         if examineAll == 1:
    #             examineAll = 0
    #         elif numChanged == 0:
    #             examineAll = 1

    #     # plt.plot(range(len(self._obj)), self._obj)
    #     # plt.show()

    #     return self

    def partial_fit(self, Sx, Sy, x, y, i):

        # Update parameters before anything else
        self.X = Sx
        self.y = Sy

        # if the instance is considered phishing
        # if y * (initial_w @ x + initial_b) * initial_Psi < model.slack:

        if y * (self.w @ x + self.b) < self.slack:

            # Initialize error cache
            self.errors = self.decision_function(self.X) - self.y

            # Track time to train
            tic = time()
            # Train model
            self.train()
            toc = time()
            
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
            self.get_clusters(2)  # should be 2, 3, or 4...
        
        # Adding instance to the window
        Sx = np.append(Sx, [x], 0)
        Sy = np.append(Sy, [y], 0)

        return Sx, Sy
        
