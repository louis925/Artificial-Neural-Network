# Copyright 2017 Louis Yang. All Rights Reserved.
'''
Artificial neural network with arbitrary number of fully connected layers and 
sigmoid activation function.

neural_network.py contain the neural network object, sigmoid function, cost 
function.
'''

import math
import numpy as np

max_float = 1e50  # max float nuber (cutoff for underflow of log)

g = lambda x: 0.5 * (1 + math.tanh(0.5 * x))  # logistic function
g_m_1 = lambda x: 0.5 * (1 - math.tanh(0.5 * x))  # 1 - logistic function
gf = np.vectorize(g)  # vectorized logistic function

def log_s(x):
    '''safe logarithm'''
    if x == 0:
        return -max_float
    else:
        return math.log(x)

def cost_f_i(y, y_pred):
    '''cost function of single data and single output'''
    if y == 1:
        return -log_s(y_pred)
    elif y == 0:
        return -log_s(1 - y_pred)
    else:
        return 0.0

cost_f = np.vectorize(cost_f_i)  # vectorized cost_f_i(y, y_pred)

class NeuralNetwork(object):

    def __init__(self, input_dim, output_dim, hidden_layers_list):
        '''hidden_layers_list: list of number of units in each hidden layers'''
        self.X = np.array([])
        self.Y = np.array([])
        self.m = 0  # total number of training data points
        self.n = input_dim
        self.yn = output_dim
        self.sl = [input_dim] + hidden_layers_list + [output_dim]  
        # number of neurons in each layer
        self.n_layers = len(self.sl) - 1  # number of input and hidden layers
        self.t = [[]] * self.n_layers  # list for parameter matrix t
        self.t0 = [[]] * self.n_layers  # list for bias vectors
        return

    def train(self, x_data, y_data, iters, step, verbose = 10, conti = False,
              norm_method = 'normal'):
        '''Train the neural network with data
        
        x_data: training data features (2D float np.array with dimension = 
                (number of examples, number of features))
        y_data: training data classes (2D np.array with values 0 or 1 
                and dimension = (number of examples, number of classes))
        iters: number of iterations or steps
        step: step size or learning rate
        verbose: number of iterations before showing training status
        conti: Bool, whether this si a continue run following previous
        '''
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
            self.m = len(y_data) # total number of training data points
        
        if conti:  # Continue run
            # Normalize X using the previous factors:
            self.X = self.rescale_X(self.X_ori)
        else:  # First run
            # Normalize X:
            self.normal_X(method = norm_method)
            
            # Initialize the t matrices and t0 vectors:
            self.init_t()
            
        # Training neural network:
        for i in range(iters):
            if i % verbose == 0:
                c = self.cost()
                print(str(i + 1) + 'th run: ' + str(c))
            self.activation_compute()
            self.delta_compute()
            self.t_grad_gen()
            self.t0_grad_gen()
            self.t_update(step)
        print('Final run: ' + str(self.cost()))
        return

    def normal_X(self, method = 'normal'):
        '''normalize X'''
        if method == 'minmax':
            X_ori_min = self.X_ori.min(0)
            X_ori_max = self.X_ori.max(0)
            self.X_ori_center = (X_ori_min + X_ori_max) / 2
            self.X_ori_scale = (X_ori_max - X_ori_min) / 2
        else:  # All other method use 'normal'
            self.X_ori_center = self.X_ori.mean(0)
            self.X_ori_scale = self.X_ori.std(0)
        self.X_ori_scale = np.array([1.0 if x == 0.0 else x for x in self.X_ori_scale])  # avoid divid by zero issue
        self.X = self.rescale_X(self.X_ori)
        return
    
    def rescale_X(self, X_ori):
        '''rescale X using the factor stored in the network'''
        return (X_ori - self.X_ori_center) / self.X_ori_scale
    
    def unrescale_X(self, X):
        '''undo the rescale of X using the factor stored in the network'''
        return X * self.X_ori_scale + self.X_ori_center
    
    def init_t(self):
        '''assign random initial values to the parameter matrices t in the 
        range of tmin to tmax
        '''
        self.ti_epsi = np.array([np.sqrt(6.0/(self.sl[l] + self.sl[l+1])) \
                                 for l in range(self.n_layers)])
        for l in range(self.n_layers):
            self.t[l] = self.ti_epsi[l] * (np.random.rand(self.sl[l+1],
                  self.sl[l]) - 0.5)
            self.t0[l] = self.ti_epsi[l] * (np.random.rand(self.sl[l+1]) - 0.5)
        return

    def y_pred_ori(self, x_ori):
        '''input a single data x (original), return the prediction y'''
        x_next = self.rescale_X(x_ori)        
        for j in range(self.n_layers):
            #x_next = gf(np.dot(self.t[j], x_next) + self.t0[j])
            x_next = gf(np.inner(x_next, self.t[j]) + self.t0[j])
        return x_next

    def y_pred(self, x):
        '''input a single data x (normalized), return the prediction y'''
        x_next = x  # single data feature x
        for j in range(self.n_layers):
            #x_next = gf(np.dot(self.t[j], x_next) + self.t0[j])
            x_next = gf(np.inner(x_next, self.t[j]) + self.t0[j])
        return x_next
    
    def Y_pred_gen(self):
        '''return the list of prediction y from the training data x_data'''
        # self.Y_pred = np.array(map(self.y_pred, self.X))
        self.Y_pred = np.apply_along_axis(self.y_pred, 1, self.X)
        return self.Y_pred

    def Chi_square(self):
        '''total Chi square error'''
        return np.sum(( self.Y_pred - self.Y )**2)

    def cost(self):
        '''return the total cost according to the training data 
        for the neural network
        '''
        self.Y_pred_gen()
        return np.sum(cost_f(self.Y, self.Y_pred)) / self.m

    def activation_compute(self):
        '''compute the activation of each layers based on data (forward
        propagation)
        '''
        self.A = [self.X] + [[]] * self.n_layers
        for l in range(self.n_layers):
            self.A[l + 1] = gf(np.tensordot(self.A[l], self.t[l], (1,1))
                               + self.t0[l])
        return

    def delta_compute(self):
        '''compute the delta of each layers based on data (backward
        propagation)
        '''
        self.delta = [[]] * self.n_layers
        self.delta[self.n_layers - 1] = self.Y_pred - self.Y
        for l in range(self.n_layers - 1, 0, -1):
            self.delta[l - 1] = np.dot(self.delta[l], self.t[l]) * self.A[l] \
                                * (1 - self.A[l])
        return

    def t_grad_gen(self):
        '''compute the derivative of the cost function w.r.t. t matrix'''
        self.t_grad = [[]] * self.n_layers
        for l in range(self.n_layers):
            self.t_grad[l] = np.tensordot(self.delta[l], self.A[l], (0, 0)) \
                             / self.m
        return

    def t0_grad_gen(self):
        '''compute the derivative of the cost function w.r.t. t0 vector'''
        self.t0_grad = [0] * self.n_layers
        for l in range(self.n_layers):
            self.t0_grad[l] = np.sum(self.delta[l], 0) / self.m
        return

    def t_update(self, step):
        '''update t matrices and t0 vectors according to gradient descent'''
        for l in range(self.n_layers):
            self.t[l] -= step * self.t_grad[l]
            self.t0[l] -= step * self.t0_grad[l]
        return

    def predict(self, X_ori):
        '''Given data set X_ori (original) return the prediction of X_ori'''
        Y_pred_float = np.apply_along_axis(self.y_pred_ori, 1, X_ori)
        return 1 * (Y_pred_float > 0.5)
    
    def predict_class(self, X_ori):
        '''Return predicted class index of data set X_ori'''
        X = self.rescale_X(X_ori)
        return np.argmax(np.apply_along_axis(self.y_pred, 1, X), axis=1)
    
    def accuracy_regression(self, X_ori, Y):
        '''Regression accuracy based on given data set X and Y
        '''
        Y = np.array(Y)
        return np.sum(Y == self.predict(X_ori)) / Y.size

    def accuracy_class(self, X_ori, Y):
        '''Classification accuracy based on given data set X and Y
        '''
        Y_class = to_class(Y)  # extract the class index
        Y_class_pred = self.predict_class(X_ori)
        return np.sum(Y_class == Y_class_pred) / len(Y_class)

def to_class(Y):
    ''' Convert class score to class index
    Given Y = (0,0,..,1,0,0,...), return the index of the first 1
    Given Y = (0.1, 0.9, 0.2,..), return the index of the most likely class
    '''
    return np.argmax(Y, axis=-1)