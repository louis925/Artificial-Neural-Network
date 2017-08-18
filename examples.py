# Copyright 2017 Louis Yang. All Rights Reserved.
'''
Artificial neural network with arbitrary number of fully connected layers and 
sigmoid activation function.

examples.py contains all the test case examples
'''
import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn
# %%
#test cases
def test0():
    x = [[1],[2],[3],[4]]
    y = [[0],[0],[1],[1]]
    sl = []  # no hidden layer
    model = nn.NeuralNetwork(len(x[0]), len(y[0]), sl)
    model.train(x,y,100,0.3)
    print('Predict:', model.predict(model.X_ori))
    print('Cost:', model.cost())
    print('X^2:', model.Chi_square())
    print('Regression Accuracy:', model.accuracy_regression(model.X_ori, 
                                                            model.Y))
    # Should be 1.0
    return model
# %%
def test1():
    x = [[1.0,1.0,1.0], [2.0,2.0,2.0], [3.0,3.0,3.0], [4.0,4.0,4.0]]
    y = [[1,0], [0,1], [1,1], [0,0]]
    sl = [5,3]  # 2 hidden layers
    model = nn.NeuralNetwork(len(x[0]), len(y[0]), sl)
    model.train(x,y,100,0.3)
    print('Predict:', model.predict(model.X_ori))
    print('Regression Accuracy:', model.accuracy_regression(model.X_ori, 
                                                            model.Y))
    # Should be 0.75~1.0
    return model
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
    hofx = nn.gf(x)
    x = x * a1 + a0
    y = np.array([1 if randsample[i] < hofx[i] else 0 for i in range(n)])
    #y = np.array([[1] if yi > 0.5 else [0] for yi in y_log])
    plt.style.use('default')
    plt.plot(x, y, '+')
    plt.plot(x, hofx)
    plt.plot(x, randsample, 'r*')
    plt.show()

    x = np.transpose([x])
    y = np.transpose([y])

    sl = []  # no hidden layer
    runs = 500
    step = 0.5
    model = nn.NeuralNetwork(len(x[0]), len(y[0]), sl)
    model.train(x, y, runs, step, optimizer='adam')

    plt.plot(x, hofx)
    plt.plot(model.X_ori, model.Y_pred, '+')
    plt.show()
    
    print('Predict:', model.predict(model.X_ori))
    print('Cost:', model.cost())
    print('X^2:', model.Chi_square())
    print('Regression Accuracy:', model.accuracy_regression(model.X_ori, 
                                                            model.Y))
    # Should be ~0.82
    return model
# %%
def test3():
    '''test the neural network as a XNOR logic unit'''
    print('Test the neural network as a XNOR logic unit')
    
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[1], [0], [0], [1]])
    
    # In principle, one should be able to produce a XNOR logic unit using only
    # 1 hidden layer with 2 units.
    sl = [3] # 1 hidden layer with 3 units (2 units is possible but not stable)
    runs = 50
    step = 0.5
    model = nn.NeuralNetwork(len(x[0]), len(y[0]), sl)
    model.train(x, y, runs, step, verbose=5, optimizer='adam',initializer='Xavier_normal')
    print('t matrices:')
    print(model.t)
    print('t0 vectors:')
    print(model.t0)
    
    print('Predict:', model.predict(model.X_ori))
    print('Cost:', model.cost())
    print('X^2:', model.Chi_square())
    print('Regression Accuracy:', model.accuracy_regression(model.X_ori, 
                                                            model.Y))
    # Should be 1.0 for most of the time
    return model
# %%
def test4():
    '''improved test of the neural network as a XNOR logic unit'''
    n = 100
    xmax = 10
    x = np.array([[0.0, 0.0]] * n)
    y = np.array([[0]] * n)
    np.random.seed(817)  # fix the initial seed for comparison purpose
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
    plt.plot([-xmax/2, xmax/2],[0, 0])
    plt.plot([0, 0],[-xmax/2, xmax/2])
    plt.show()
    
    #### Train the model ####
    sl = [4]  # 1 hidden layer with 4 neurons
    runs = 4000
    step = 0.0015
    model = nn.NeuralNetwork(len(x[0]), len(y[0]), sl)
    model.train(x, y, runs, step, verbose=200, optimizer='adam')
    print('t matrices:')
    print(model.t)
    print('t0 vectors:')
    print(model.t0)
    
    Y_pred_dis = model.predict(model.X_ori)  # discrete predicted result
    X_pred_p = []  # points classified as positive
    X_pred_n = []  # points classified as negative
    for i in range(len(Y_pred_dis)):
        if Y_pred_dis[i, 0] == 1:
            X_pred_p.append(model.X_ori[i])
        else:
            X_pred_n.append(model.X_ori[i])
    X_pred_p = np.array(X_pred_p)
    X_pred_n = np.array(X_pred_n)
    plt.plot(X_pred_p[:, 0], X_pred_p[:, 1], 'o')
    plt.plot(X_pred_n[:, 0], X_pred_n[:, 1], 'x')
    plt.plot([-xmax/2, xmax/2],[0, 0])
    plt.plot([0, 0],[-xmax/2, xmax/2])
    plt.show()
    
    print('Correct:', model.Y.T)
    print('Predict:', Y_pred_dis.T)
    print('Cost:', model.cost())
    print('X^2:', model.Chi_square())
    print('Regression Accuracy:', model.accuracy_regression(model.X_ori, 
                                                            model.Y))
    # Should be 0.92
    return model
# %%
def test_class_0():
    x = [[1, 0], [1, 1], [0, 0], [0, -1], [2, 1], [2, 0]]
    y = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    sl = [4]  # 1 hidden layers
    model = nn.NeuralNetwork(len(x[0]), len(y[0]), sl)
    model.train(x,y,100,0.7)
    print('Prediction for training data:')
    print(model.predict(model.X_ori))
    print('Predicted class index:', model.predict_class(model.X_ori))
    print('Cost:', model.cost())
    print('X^2:', model.Chi_square())  # Should be ~e-06
    print('Regression accuracy:', model.accuracy_regression(model.X_ori, model.Y))
    print('Classification accuracy:', model.accuracy_class(model.X_ori, model.Y))
    # Both should be 1.0
    return model
