'''
    Author: Victor Barboza

    Assuming every slide of the window will update X, y, b, w, slack, 
    and maybe alphas and errors
'''
from SVM_w_SMO import *

class AAOSVM(SMOModel):
    '''
        Reproduction of the Adversary-Aware Online SVM
    '''
    def __init__(self, X, y, alphas, errors, C=100, m=100, Gp=250, Em=1, Er=1, Ym=1, Yr=1, s=0.6, kernel='linear'):
        SMOModel.__init__(self, X, y, C, alphas, 0, errors, kernel)
        self.mem = m                            # memory parameter
        self.Gp = Gp                            # period to approximate sequential equilibrium strategies
        self.Em = Em                            # reward for classifying malicious correctly
        self.Er = Er                            # reward for classifying regular correctly
        self.Ym = Ym                            # punishment for misclassifying malicious
        self.Yr = Yr                            # punishment for misclassifying regular
        self.w = np.zeros((len(X[0]),))         # weight vector
        # self.slack = np.ones((len(X[0]),)) * s  # slack vector
        self.slack = s


    def psi(self, x):
        '''
        Helper function from Psi(x)
        '''
        pass

    
    def Psi(self, x):
        '''
        Function that incorporates prior knowledge into the SVM
        '''
        return (1 + psi(x))/(self.w.T * np.ones((len(self.X[0]),)) + 2 * self.b)


    def update_weights(self, i1, i2, a1, a2):
        '''
        Update the value of a couple of weights?
        '''
        # TODO: try this
        # return (y * Psi(x)) * a @ x.T
        # TODO: change the vector multiplication to instances?
        self.w = self.w + self.y[i1] * (a1 - self.a[i1]) * self.X[i1] + self.y[i2] * (a2 - self.a[i2]) * self.X[i2]


    # def calculate_weights(self):
    #     '''
    #     Initializes the weights
    #     '''
    #     return 


    # Objective function to optimize
    def objective_function(self, alphas):
        """Returns the SVM objective function based in the input model defined by:
        `alphas`: vector of Lagrange multipliers
        `target`: vector of class labels (-1 or 1) for training data
        `kernel`: kernel function
        `X_train`: training data for model."""
        
        return np.sum(alphas) - 0.5 * np.sum((self.y[:, None] * self.y[None, :]) * self.kernel(self.X, self.X) * (alphas[:, None] * alphas[None, :]))


    # Decision function (AKA constraint(s))
    def decision_function(self, x_test):
        """Applies the SVM decision function to the input feature vectors in `x_test`."""
        
        return (self.alphas * self.y) @ self.kernel(self.X, x_test) - self.b

