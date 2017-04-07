import math
import numpy as np
import matplotlib.pyplot as plt

max_float = 1e50 #max float nuber (cutoff for underflow of log)

g = lambda x: 0.5 * (1 + math.tanh(0.5 * x)) #logistic function
g_m_1 = lambda x: 0.5 * (1 - math.tanh(0.5 * x)) #1 - logistic function
gf = np.vectorize(g) #vectorized logistic function

def log_s(x):
    'safe logarithm'
    if x == 0:
        return -max_float
    else:
        return math.log(x)

def cost_f_i(y, y_pred):
    'cost function of single data and single output'
    if y == 1:
        return -log_s(y_pred)
    elif y == 0:
        return -log_s(1 - y_pred)
    else:
        return 0.0

cost_f = np.vectorize(cost_f_i)

class NeuralNetwork(object):

    def __init__(self, input_dim, output_dim, hidden_layers_list):
        'hidden_layers_list: list of number of units in each hidden layers'
        self.X = np.array([])
        self.Y = np.array([])
        self.m = 0 #total number of training data points
        self.n = input_dim
        self.yn = output_dim
        self.sl = [input_dim] + hidden_layers_list + [output_dim] #number of neurons in each layer
        self.n_layers = len(self.sl) - 1 #number of input and hidden layers
        self.t = [[]] * self.n_layers #list for parameter matrix t
        self.t0 = [[]] * self.n_layers #list for bias vectors
        return

    def train(self, x_data, y_data, iters, step):
        'x_data: training data inputs feature,\n \
        y_data: training data output list'
        if len(x_data[0]) != self.n:
            print('inconsistent dim of x_data')
            return
        if len(y_data[0]) != self.yn:
            print('inconsistent dim of y_data')
            return
        self.X_ori = np.array(x_data)
        self.Y = np.array(y_data)

        if len(x_data) != len(y_data):
            print('inconsistent number of data')
            self.m = min(len(x_data),len(y_data))
        else:
            self.m = len(y_data) #total number of training data points

        #Normalize X:
        self.normal_X()
        
        #Initialize the t matrix:
        self.init_t()
        
        #Complete the code for training neural network:
        for i in xrange(iters):
            if i % 10 == 0:
                c = self.cost()
                print(str(i + 1) + 'th run: ' + str(c))
            self.activation_compute()
            self.delta_compute()
            self.t_grad_gen()
            self.t0_grad_gen()
            self.t_update(step)
        print('Final run: ' + str(self.cost()))
        return

    def normal_X(self):
        'normalize X'
        self.X_ori_mean = self.X_ori.mean(0)
        self.X_ori_std = self.X_ori.std(0)
        self.X = self.X_ori
        self.X = self.X - self.X_ori_mean
        self.X = self.X / self.X_ori_std
        return
    
    def init_t(self):
        'assign random initial values to the parameter matrices t in the range of tmin to tmax'
        self.ti_epsi = np.array([np.sqrt(6.0/(self.sl[l] + self.sl[l+1])) for l in xrange(self.n_layers)])
        for l in xrange(self.n_layers):
            self.t[l] = self.ti_epsi[l] * (np.random.rand(self.sl[l + 1], self.sl[l]) - 0.5)
            self.t0[l] = self.ti_epsi[l] * (np.random.rand(self.sl[l + 1]) - 0.5)
        return

    def y_pred_ori(self, x_ori):
        'input a single data x (original), return the prediction y'
        x_next = (x_ori - self.X_ori_mean) / self.X_ori_std #normalized
        for j in xrange(self.n_layers):
            x_next = gf(np.dot(self.t[j], x_next) + self.t0[j])
        return x_next

    def y_pred(self, x):
        'input a single data x (normalized), return the prediction y'
        x_next = x #single data feature x
        for j in xrange(self.n_layers):
            x_next = gf(np.dot(self.t[j], x_next) + self.t0[j])
        return x_next
    
    def Y_pred_gen(self):
        'return the list of prediction y from the training data x_data'
        self.Y_pred = np.array(map(self.y_pred, self.X))
        return self.Y_pred

    def Chi_square(self):
        'total Chi square error'
        return np.sum(( self.Y_pred - self.Y )**2)

    def cost(self):
        'return the total cost according to the training data for the neural network'
        self.Y_pred_gen()
        return np.sum(cost_f(self.Y, self.Y_pred)) / self.m

    def activation_compute(self):
        'compute the activation of each layers based on data'
        self.A = [self.X] + [[]] * self.n_layers
        for l in xrange(self.n_layers):
            self.A[l + 1] = gf(np.tensordot(self.A[l], self.t[l], (1,1)) + self.t0[l])
        return

    def delta_compute(self):
        'compute the delta of each layers based on data'
        self.delta = [[]] * self.n_layers
        self.delta[self.n_layers - 1] = self.Y_pred - self.Y
        for l in xrange(self.n_layers - 1, 0, -1):
            self.delta[l - 1] = np.dot(self.delta[l], self.t[l]) * self.A[l] * (1 - self.A[l])
        return

    def t_grad_gen(self):
        'compute the derivative of the cost function w.r.t. t matrix'
        self.t_grad = [[]] * self.n_layers
        for l in xrange(self.n_layers):
            self.t_grad[l] = np.tensordot(self.delta[l], self.A[l], (0, 0)) / self.m
        return

    def t0_grad_gen(self):
        'compute the derivative of the cost function w.r.t. t0 vector'
        self.t0_grad = [0] * self.n_layers
        for l in xrange(self.n_layers):
            self.t0_grad[l] = np.sum(self.delta[l], 0) / self.m
        return

    def t_update(self, step):
        'update t matrices and t0 vectors according to gradient descent'
        for l in xrange(self.n_layers):
            self.t[l] -= step * self.t_grad[l]
            self.t0[l] -= step * self.t0_grad[l]
        return
    
#test cases
def test0():
    sl = []
    x = [[1],[2],[3],[4]]
    y = [[0],[0],[1],[1]]
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x,y,100,0.3)
    return nn

def test1():
    x = [[1.0,1.0,1.0], [2.0,2.0,2.0], [3.0,3.0,3.0], [4.0,4.0,4.0]]
    y = [[1,0], [0,1], [1,1], [0,0]]
    sl = [5,3]
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x,y,100,0.3)
    return nn

def test2():
    'test the neural network as a logistic function fitting'
    n = 100
    a0 = 10.0
    a1 = 5.0
    xmax = 10
    sl = []
    randsample = np.random.rand(n)
    x = (np.arange(n)/float(n) - 0.5) * xmax
    #x = a1 * (np.random.rand(n, 1) - 0.5) + a0
    hofx = gf(x)
    x = x * a1 + a0
    y = np.array([1 if randsample[i] < hofx[i] else 0 for i in xrange(n)])
    #y = np.array([[1] if yi > 0.5 else [0] for yi in y_log])
    plt.plot(x, y, '+')
    plt.plot(x, hofx)
    plt.plot(x, randsample, 'r*')
    plt.show()

    x = np.transpose([x])
    y = np.transpose([y])

    runs = 500
    step = 0.5
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x, y, runs, step)

    plt.plot(x, hofx)
    plt.plot(nn.X_ori, nn.Y_pred, '+')
    plt.show()
    
    return nn

def test3():
    'test the neural network as a XNOR logic unit'
    
    sl = [2]
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[1], [0], [0], [1]])
    
    runs = 500
    step = 0.5
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x, y, runs, step)
    print('t matrices:')
    print(nn.t)
    print('t0 vectors:')
    print(nn.t0)
    return nn

def test4():
    'improved test of the neural network as a XNOR logic unit'
    sl = [2]

    n = 100
    xmax = 10
    x = np.array([[0.0, 0.0]] * n)
    y = np.array([[0]] * n)
    for i in xrange(n):
        a1 = xmax * (np.random.rand() - 0.5)
        a2 = xmax * (np.random.rand() - 0.5)
        if a1 * a2 > 0:
            y[i, 0] = 1
        else:
            y[i, 0] = 0
        x[i, 0] = a1
        x[i, 1] = a2

    n_p = np.sum(y) #number of positive data points

    x1p = np.array([0.0] * n_p)
    x2p = np.array([0.0] * n_p)

    x1n = np.array([0.0] * (n - n_p))
    x2n = np.array([0.0] * (n - n_p))
    
    i_p = 0
    i_n = 0
    for i in xrange(n):
        if y[i, 0] == 1:
            x1p[i_p] = x[i, 0]
            x2p[i_p] = x[i, 1]
            i_p += 1
        else:
            x1n[i_n] = x[i, 0]
            x2n[i_n] = x[i, 1]
            i_n += 1

    plt.plot(x1p, x2p, 'o')
    plt.plot(x1n, x2n, 'x')
    plt.show()
    
    runs = 500
    step = 0.2
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x, y, runs, step)
    print('t matrices:')
    print(nn.t)
    print('t0 vectors:')
    print(nn.t0)
    return nn









