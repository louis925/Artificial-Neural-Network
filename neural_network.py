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
        self.m = 0  # total number of training samples
        self.n = input_dim
        self.yn = output_dim
        self.sl = [input_dim] + hidden_layers_list + [output_dim]  
        # number of neurons in each layer
        self.n_layers = len(self.sl) - 1  # number of input and hidden layers
        self.t = [[]] * self.n_layers  # list for parameter matrix t
        self.t0 = [[]] * self.n_layers  # list for bias vectors
        return

    def train(self, x_data, y_data, iters, step, verbose = 10, conti = False,
              norm_method = 'normal', optimizer = 'adam', 
              initializer = 'Xavier_uniform'):
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
        
         # Recording parameters
        self.step = step
        self.norm_method = norm_method
        self.optimizer = optimizer
        self.initializer = initializer
        
        # Initialization or not
        if conti:  # Continue run
            # normalize X using the previous factors:
            self.X = self.rescale_X(self.X_ori)
        else:  # First run
            # new normalization from X:
            self.normal_X(method = norm_method)
            
            # Initialize the t matrices and t0 vectors:
            self.init_t(initializer)
            
            self.iter = 1  # Set total iteration to 1
            if optimizer == 'sgd':
                pass
            else:
                self.init_adam()  # initialization for adam method
                            
        # Training neural network:
        self.Y_pred_gen()  # For computing initial cost()
        for i in range(iters):
            if i % verbose == 0:
                c = self.cost()
                print(str(i) + 'th run: ' + str(c))
            self.activation_compute()
            self.delta_compute()
            self.t_grad_gen()
            self.t0_grad_gen()
            if optimizer == 'sgd':
                self.t_update_sgd(step)
            else:  # default using adams
                self.t_update_adam(step, self.iter)
            self.iter += 1
        print('Final run: ' + str(self.cost()))
        return

    def normal_X(self, method = 'normal'):
        '''normalize X'''
        if method == 'minmax':
            X_ori_min = self.X_ori.min(0)
            X_ori_max = self.X_ori.max(0)
            self.X_ori_center = (X_ori_min + X_ori_max) / 2
            self.X_ori_scale = (X_ori_max - X_ori_min) / 2
            self.X_ori_scale = np.array([1.0 if x == 0.0 else x for x in \
                                         self.X_ori_scale])
        elif method == 'minmax_all':
            X_ori_min = self.X_ori.min()
            X_ori_max = self.X_ori.max()
            self.X_ori_center = (X_ori_min + X_ori_max) / 2
            self.X_ori_scale = (X_ori_max - X_ori_min) / 2
            if self.X_ori_scale == 0.0: self.X_ori_scale = 1.0
        else:  # All other method use 'normal'
            self.X_ori_center = self.X_ori.mean(0)
            self.X_ori_scale = self.X_ori.std(0)
            self.X_ori_scale = np.array([1.0 if x == 0.0 else x for x in \
                                         self.X_ori_scale])
        # avoid divid by zero issue
        self.X = self.rescale_X(self.X_ori)
        return
    
    def rescale_X(self, X_ori):
        '''rescale X using the factor stored in the network'''
        return (X_ori - self.X_ori_center) / self.X_ori_scale
    
    def unrescale_X(self, X):
        '''undo the rescale of X using the factor stored in the network'''
        return X * self.X_ori_scale + self.X_ori_center
    
    def init_t(self, initializer='Xavier_uniform'):
        '''assign random initial values to the parameter matrices t and vector 
        t0 with Xavier initialization.
        See https://keras.io/initializers/
        '''
        if initializer == 'He_uniform':
            self.ti_epsi = np.array([np.sqrt(6.0/(self.sl[l])) \
                                     for l in range(self.n_layers)])
        elif initializer == 'He_normal':
            self.ti_epsi = np.array([np.sqrt(2.0/(self.sl[l])) \
                                     for l in range(self.n_layers)])
        elif initializer == 'Xavier_normal':
            self.ti_epsi = np.array([np.sqrt(2.0/(self.sl[l] + self.sl[l+1])) \
                                     for l in range(self.n_layers)])
        else:  # 'Xavier_uniform'
            self.ti_epsi = np.array([np.sqrt(6.0/(self.sl[l] + self.sl[l+1])) \
                                     for l in range(self.n_layers)])
        
        for l in range(self.n_layers):
            self.t[l] = self.ti_epsi[l] * (2.0 * np.random.rand(self.sl[l+1],
                  self.sl[l]) - 1.0)
            #self.t0[l] = self.ti_epsi[l] * (2.0 * np.random.rand(self.sl[l+1]) - 1.0)
            self.t0[l] = np.zeros(self.sl[l+1])
        return
    
    def init_adam(self):
        '''Initialization for the m and v parameters in Adam method'''
        self.adam_m = [np.zeros(layer.shape) for layer in self.t]
        self.adam_m0 = [np.zeros(layer0.shape) for layer0 in self.t0]
        self.adam_v = [np.zeros(layer.shape) for layer in self.t]
        self.adam_v0 = [np.zeros(layer0.shape) for layer0 in self.t0]
        return
    
    def y_pred_ori(self, x_ori):
        '''input data x_ori (original), return the prediction y
        x_ori can be single or a list samples
        '''
        return self.y_pred(self.rescale_X(x_ori))

    def y_pred(self, x):
        '''input data x (normalized), return the prediction y
        x can be single or a list samples
        '''
        for l in range(self.n_layers):
            x = gf(np.inner(x, self.t[l]) + self.t0[l])
        return x
    
    def Y_pred_gen(self):
        '''Generate and return the prediction Y from the training data X'''
        self.Y_pred = self.y_pred(self.X)
        return self.Y_pred

    def activation_compute(self):
        '''compute the activation of each layers based on data (forward
        propagation)
        '''
        self.A = [self.X] + [[]] * self.n_layers
        for l in range(self.n_layers):
            #self.A[l + 1] = gf(np.tensordot(self.A[l], self.t[l], (1,1)) + self.t0[l])
            self.A[l + 1] = gf(np.inner(self.A[l], self.t[l]) + self.t0[l])
        self.Y_pred = self.A[-1]
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

    def t_update_sgd(self, step):
        '''update t matrices and t0 vectors using gradient descent'''
        for l in range(self.n_layers):
            self.t[l] -= step * self.t_grad[l]
            self.t0[l] -= step * self.t0_grad[l]
        return

    def t_update_adam(self, step, i, eps=1e-8, beta1=0.9, beta2=0.999):
        '''Update t matrices and t0 vectors using the Adam method
        Ref: https://arxiv.org/abs/1412.6980
        '''
        mfactor = 1.0 - beta1 ** i
        vfactor = 1.0 - beta2 ** i
        mibeta1 = 1.0 - beta1
        mibeta2 = 1.0 - beta2
        for l in range(self.n_layers):
            self.adam_m[l] = beta1*self.adam_m[l] + mibeta1 * self.t_grad[l]
            self.adam_m0[l] = beta1*self.adam_m0[l] + mibeta1 * self.t0_grad[l]
            mt = self.adam_m[l] / mfactor
            mt0 = self.adam_m0[l] / mfactor
            self.adam_v[l] = beta2 * self.adam_v[l] + \
                             mibeta2 * (self.t_grad[l] ** 2)
            self.adam_v0[l] = beta2 * self.adam_v0[l] + \
                              mibeta2 * (self.t0_grad[l] ** 2)
            vt = self.adam_v[l] / vfactor
            vt0 = self.adam_v0[l] / vfactor
            
            self.t[l] -= step * mt / (np.sqrt(vt) + eps)
            self.t0[l] -= step * mt0 / (np.sqrt(vt0) + eps)
        return
    
    def Chi_square(self):
        '''total Chi square error
        Run activation_compute
        '''
        return np.sum(( self.Y_pred - self.Y )**2)

    def cost(self):
        '''return the total cost according to the training data 
        for the neural network
        '''
        #self.Y_pred_gen()  # Need to remove this
        return np.sum(cost_f(self.Y, self.Y_pred)) / self.m

    def predict(self, X_ori):
        '''Return the 0 or 1 prediction of X_ori in each Y'''
        #Y_pred_float = np.apply_along_axis(self.y_pred_ori, 1, X_ori)
        #Y_pred_float = self.y_pred_ori(X_ori)
        #return 1 * (Y_pred_float > 0.5)
        return 1 * (self.y_pred_ori(X_ori) > 0.5)    
    
    def predict_class(self, X_ori):
        '''Return predicted class index of data set X_ori'''
        #X = self.rescale_X(X_ori)
        #return np.argmax(np.apply_along_axis(self.y_pred, 1, X), axis=1)
        return to_class(self.y_pred_ori(X_ori))
    
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
    
    def train_accuracy_reg(self):
        '''Training regression accuracy'''
        return self.accuracy_regression(self.X_ori, self.Y)
    
    def train_accuracy_class(self):
        '''Training classification accuracy'''
        return self.accuracy_class(self.X_ori, self.Y)

def to_class(Y):
    '''Convert class score to class index
    Given Y = (0,0,..,1,0,0,...), return the index of the first 1
    Given Y = (0.1, 0.9, 0.2,..), return the index of the most likely class
    '''
    return np.argmax(Y, axis=-1)

def to_Y(class_index_set, n_class):
    '''Convert class index set into Y = (0,0,...1,0,...) where the 1 locates at
    the class index.
    '''
    return np.array([[1 if class_index == i else 0 for i in range(n_class)]
                      for class_index in class_index_set])