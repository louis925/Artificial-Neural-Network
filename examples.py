# Copyright 2017 Louis Yang. All Rights Reserved.
'''
Artificial neural network with arbitrary number of fully connected layers and 
sigmoid activation function.

examples.py contains all the test case examples
'''
from neural_network import *
import numpy as np
import matplotlib.pyplot as plt
# %%
#test cases
def test0():
    x = [[1],[2],[3],[4]]
    y = [[0],[0],[1],[1]]
    sl = []  # no hidden layer
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x,y,100,0.3)
    return nn
# %%
def test1():
    x = [[1.0,1.0,1.0], [2.0,2.0,2.0], [3.0,3.0,3.0], [4.0,4.0,4.0]]
    y = [[1,0], [0,1], [1,1], [0,0]]
    sl = [5,3]  # 2 hidden layers
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x,y,100,0.3)
    return nn
# %%
def test2():
    '''test the neural network as a logistic function fitting'''
    print('Test the neural network as a logistic function fitting:')
    n = 100
    a0 = 10.0
    a1 = 5.0
    xmax = 10
    randsample = np.random.rand(n)
    x = (np.arange(n)/float(n) - 0.5) * xmax
    #x = a1 * (np.random.rand(n, 1) - 0.5) + a0
    hofx = gf(x)
    x = x * a1 + a0
    y = np.array([1 if randsample[i] < hofx[i] else 0 for i in range(n)])
    #y = np.array([[1] if yi > 0.5 else [0] for yi in y_log])
    plt.plot(x, y, '+')
    plt.plot(x, hofx)
    plt.plot(x, randsample, 'r*')
    plt.show()

    x = np.transpose([x])
    y = np.transpose([y])

    sl = []  # no hidden layer
    runs = 500
    step = 0.5
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x, y, runs, step)

    plt.plot(x, hofx)
    plt.plot(nn.X_ori, nn.Y_pred, '+')
    plt.show()
    
    return nn
# %%
def test3():
    '''test the neural network as a XNOR logic unit'''
    print('Test the neural network as a XNOR logic unit')
    
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[1], [0], [0], [1]])
    
    sl = [2] #1 hidden layer with 2 units
    runs = 500
    step = 0.5
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x, y, runs, step)
    print('t matrices:')
    print(nn.t)
    print('t0 vectors:')
    print(nn.t0)
    return nn
# %%
def test4():
    'improved test of the neural network as a XNOR logic unit'
    n = 100
    xmax = 10
    x = np.array([[0.0, 0.0]] * n)
    y = np.array([[0]] * n)
    for i in range(n):
        a1 = xmax * (np.random.rand() - 0.5)
        a2 = xmax * (np.random.rand() - 0.5)
        if a1 * a2 > 0:
            y[i, 0] = 1
        else:
            y[i, 0] = 0
        x[i, 0] = a1
        x[i, 1] = a2

    n_p = np.sum(y)  # number of positive data points

    x1p = np.array([0.0] * n_p)
    x2p = np.array([0.0] * n_p)

    x1n = np.array([0.0] * (n - n_p))
    x2n = np.array([0.0] * (n - n_p))
    
    i_p = 0
    i_n = 0
    for i in range(n):
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
    
    sl = [2]  # 1 hidden layer with 2 neurons
    runs = 500
    step = 0.2
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x, y, runs, step)
    print('t matrices:')
    print(nn.t)
    print('t0 vectors:')
    print(nn.t0)
    return nn
# %%
def test_class_0():
    x = [[1, 0], [1, 1], [0, 0], [0, -1], [2, 1], [2, 0]]
    y = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    sl = [4]  # 1 hidden layers
    nn = NeuralNetwork(len(x[0]), len(y[0]), sl)
    nn.train(x,y,300,0.7)
    print('Prediction for training data:')
    print(nn.predict(nn.X_ori))
    print('Predicted class index:', nn.predict_class(nn.X_ori))
    print('Regression accuracy:', nn.accuracy_regression(nn.X_ori, nn.Y))
    print('Classification accuracy:', nn.accuracy_class(nn.X_ori, nn.Y))
    return nn
