import numpy as np

def cost_function(x, y, theta):
    '''
    Note that this a vectorized cost function, meaning that X is our feature matrix properly augmented
    with ones. It also is expected a training set of size M and N features.
    @param x: Mx(n+1) feature matrix
    @param y: 1xM output vector
    @param theta: 1x(N+1) weight vector
    '''
    m = len(y)
    h = x@theta
    dif = np.subtract(h,y.T)
    dif = np.square(dif)
    dif = dif.sum()/(2*m)
    return dif

def partial_derivatives(x,y,theta):
    m = len(y)
    h = x@theta
    dif = np.subtract(h,y.T)
    dif = np.multiply(dif, x.T)
    dif = dif.sum(axis=1)/(m)
    return dif
	
def gradient_descent(X, y, theta, alpha, itern, debugging = False):
    '''
    Iterative Gradient descent. 
    @param X: feature matrix
    @param y: output vector
    @param theta: weights to be found
    @param alpha: learning rate
    @param itern: the number of iterations
    @paran debugging: if debugging is True, it will return a 2D array with theta vector for each iterations
    
    @out theta_opt: the theta values that minimize the cost function.
    @out theta_matrix: 2D array with theta row vectors for each iteration
    '''
    theta_matrix = []
    for i in range(itern):
        theta = theta - alpha*partial_derivatives(X,y,theta)
        theta_matrix.append(list(theta))
    if(debugging):
        return [theta, theta_matrix]
    return theta
	
def normal_equation(X, Y):
	'''
	Calculates the vector theta that minimizes the cost function
	'''
	theta = np.linalg.pinv(X.T@X)@X.T@Y
	return theta