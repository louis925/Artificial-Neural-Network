import math
import numpy as np

g = lambda x: 0.5 * (math.tanh(0.5 * x) + 1) #logistic function
gf = np.vectorize(g) #vectorized logistic function

class NeuralNetwork(object):

    def __init__(self, x_data, y_data, hidden_layers_list):
        '''x_data: training data inputs feature,\n \
        y_data: output list,\n \
        hidden_layers_list: list of number of units in each hidden layers'''
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        if len(x_data) != len(y_data):
            print('inconsistent number of data')
            self.m = min(len(x_data),len(y_data))
        else:
            self.m = len(y_data) #total number of training data points

        self.n = len(x_data[0]) #number of input features for each data

        if isinstance(self.y_data[0], np.ndarray):
            self.yn = len(y_data[0])
        else:
            self.yn = 1

        self.sl = [self.n] + hidden_layers_list + [self.yn] #number of neurons in each layer
        self.n_layers = len(self.sl) - 1 #number of input and hidden layers
        self.t = [] #list for parameter matrix t
        return

    def init_t(self, tmin = -1, tmax = 1):
        '''assign random initial values to the parameter matrices t'''
        self.tmin = tmin
        self.tmax = tmax
        for j in xrange(self.n_layers):
            self.t.append((tmax - tmin) * np.random.rand(self.sl[j + 1], self.sl[j]) + tmin)
        return
    
    def z(self, j, x):
        '''return the result vector of t[j] matrix acting on x vector'''
        return np.dot(self.t[j], x)

    def y_pred(self, x):
        '''input a data return the prediction y'''
        x_next = x #single data feature x
        for j in xrange(self.n_layers):
            x_next = gf(self.z(j, x_next))
        return x_next

    def y_pred_data(self):
        '''return the list of prediction y from the training data x_data'''
        return np.array(map(self.y_pred, self.x_data))

    def y_diff(self):
        return self.y_pred_data().flatten() - self.y_data.flatten()

    def X_2(self):
        '''total Chi square error'''
        return np.sum((self.y_pred_data().flatten() - self.y_data.flatten())**2)

#test case
x = [[1.0,1.0,1.0], [2.0,2.0,2.0],[3.0,3.0,3.0],[4.0,4.0,4.0]]
y = [1,2,3,4]
sl = [5,3]
nn = NeuralNetwork(x, y, sl)
nn.init_t()
  

  
    
