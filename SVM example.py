import cvxpy as cp

def train_svr(X,y):
    # Where did this kernel come from?
    # X = GaussianKernel1(X)
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