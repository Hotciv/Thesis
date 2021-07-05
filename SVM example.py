import pandas as pd
import cvxpy as cp

def LinearKernel(X: pd.DataFrame):
    '''
        returns: X' * X
                An NxN matrix, where N is the number of features
    '''
    return X.dot(X.T)

def GaussianKernel1(X):
    '''
        As extracted from MATLAB:
        
        % Vectorized RBF Kernel
        % This is equivalent to computing the kernel on every pair of examples
        X2 = sum(X.^2, 2);
        K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
        K = kernelFunction(1, 0) .^ K;
    '''
    pass

def train_svr(X,y):
    # Where did this kernel come from?
    # X = GaussianKernel1(X)
    X = LinearKernel(X)
    n_features = X.shape[0]
    w = cp.Variable(n_features)
    b = cp.Variable(1)
    epsilon = 0.1
    l2_square_regularization = cp.square(cp.norm(w, 2)) / 2
    constraint1 = cp.abs(y - X @ w - b) <= epsilon
    prob = cp.Problem(cp.Minimize(l2_square_regularization),constraints = [constraint1,])
    prob.solve()
    print(prob.status)
    return w.value, b.value