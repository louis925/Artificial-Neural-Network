import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn

length_x = 28 # width of the image
length_y = 28 # height of the image
n_class = 10  # number of classes
evaluate_split = 3000
output_predict = False

# %%
def read(file_path = 'example_data/MNIST/'):
    feature_train = []  # Training features
    label_train = []  # Training labels
    feature_test = []  # Testing features
        
    print('### Reading training data file...')
    with open(file_path + 'train.csv') as IFILE_TRAIN:
        for i, line in enumerate(IFILE_TRAIN):
            if i==0: continue
            data= line.strip().split(',')
            feature_train.append( [ int(d) for d in data[1:] ] )
            label_train.append( int(data[0]) )
    print("Reading complete! Totally %d data" % len(label_train))
    
    print('### Reading test data file...')
    with open(file_path + 'test.csv') as IFILE_TEST:
        for i, line in enumerate(IFILE_TEST):
            if i==0: continue
            data= line.strip().split(',')
            feature_test.append( [ int(d) for d in data ] )
            #labels_test.append( int(data[0]) )
    print("Reading complete! Totally %d data" % len(feature_test))
    
    return np.array(feature_train), np.array(label_train), np.array(feature_test)
# %% 
def output(test_predict, file_name = 'test_predict.csv'):
    '''Output result to file'''
    print('### Writing test result file to', file_name, '...')
    with open(file_name, 'w') as OFILE:
        print('ImageId,Label', file = OFILE)
        for i in range(len(test_predict)):
            print(i + 1, ',', test_predict[i], file = OFILE, sep = '')
    print("Writing result complete!")
# %%
def display(single_data, length_x = length_x, length_y = length_y):
    '''display single_data as a length_x*length_y image in grayscale'''
    plt.style.use('grayscale')
    plt.imshow(single_data.reshape(length_x, length_y))
    plt.show()
# %%
def examine(feature, label, view = 10):
    '''randomly show some example in feature and label'''
    n_feature = len(feature)
    view_list = np.random.choice(n_feature, min(view, n_feature), replace = False)
    for i in view_list:
        display(feature[i])
        print('i =',i, 'label =', label[i])        
# %%
def split_data(features, labels, n_second):
    '''
    Shuffle and slpit the data into two sets (eg. training set and evaluation 
    set).
    n_second: number of the second set
    Return: 1st set of features, 1st set of labels, 2nd set of features, and
            2nd set of labels
    '''
    n_data = len(labels)
    shuffle_index = list(range(n_data))
    np.random.shuffle(shuffle_index)
    features_shuff = np.array([features[i] for i in shuffle_index])
    labels_shuff = np.array([labels[i] for i in shuffle_index])
    return (features_shuff[n_second:], labels_shuff[n_second:], 
            features_shuff[:n_second], labels_shuff[:n_second])
# %% Main
if __name__ == '__main__':
    print('===== Test on MNIST =====')

    ###### Reading data ######
    (feature_set, label_set, feature_test) = read()

    ###### Print shapes ######
    print('--- Data dimension ---')
    print('feature set :',feature_set.shape)
    print('label   set :',label_set.shape)
    print('feature test:',feature_test.shape)
    #print(feature_train[1,])
    #print(label_train[1])
    #pd.display(feature_train[1], length_x, length_y)
    
    ###### Split into train and evaluation sets ######
    feature_train, label_train, feature_eval, label_eval = \
        split_data(feature_set, label_set, evaluate_split)
    print('training   samples:', len(label_train))
    print('evaluation samples:', len(label_eval))
    
    X_ori_train = feature_train
    Y_train = nn.to_Y(label_train, n_class)
    X_ori_eval = feature_eval
    Y_eval = nn.to_Y(label_eval, n_class)
    
    # %% [50]
    print('sl = [50]')
    sl = [50]  # hidden layer structure
    runs = 200
    step = 0.1
    model = nn.NeuralNetwork(len(X_ori_train[0]), len(Y_train[0]), sl)
    print('### Start training ...')
    # Normal work much better than minmax
    
    # %% Norm He_u
    print('Norm He_u')
    model.train(X_ori_train, Y_train, runs, step, verbose=max(int(runs/20),1), 
                norm_method='normal', initializer='He_uniform')
    print('Training   accuracy:', model.accuracy_class(X_ori_train, Y_train))
    print('Evaluation accuracy:', model.accuracy_class(X_ori_eval, Y_eval))
    label_test = model.predict_class(feature_test)
    output(model.predict_class(feature_test), 
           'example_data/MNIST/test_predict - [50] - 200 - norm - He_u.csv')
    # 0.9648, 0.924
    # %% Norm Xa_u
    print('Norm Xa_u')
    model.train(X_ori_train, Y_train, runs, step, verbose=max(int(runs/20),1), 
                norm_method='normal', initializer='Xavier_uniform')
    print('Training   accuracy:', model.accuracy_class(X_ori_train, Y_train))
    print('Evaluation accuracy:', model.accuracy_class(X_ori_eval, Y_eval))
    label_test = model.predict_class(feature_test)
    output(model.predict_class(feature_test), 
           'example_data/MNIST/test_predict - [50] - 200 - norm - Xa_u.csv')
    # 0.9665, 0.9246
    # %% Minmax_all He_u
    print('Minmax_all He_u')
    model.train(X_ori_train, Y_train, runs, step, verbose=max(int(runs/20),1), 
                norm_method='minmax_all', initializer='He_uniform')
    print('Training   accuracy:', model.accuracy_class(X_ori_train, Y_train))
    print('Evaluation accuracy:', model.accuracy_class(X_ori_eval, Y_eval))
    label_test = model.predict_class(feature_test)
    output(model.predict_class(feature_test), 
           'example_data/MNIST/test_predict - [50] - 200 - minmax_a - He_u.csv')
    #     
    # %% Minmax_all Xa_u
    print('Minmax_all Xa_u')
    model.train(X_ori_train, Y_train, runs, step, verbose=max(int(runs/20),1), 
                norm_method='minmax_all', initializer='Xavier_uniform')
    print('Training   accuracy:', model.accuracy_class(X_ori_train, Y_train))
    print('Evaluation accuracy:', model.accuracy_class(X_ori_eval, Y_eval))
    label_test = model.predict_class(feature_test)
    output(model.predict_class(feature_test), 
           'example_data/MNIST/test_predict - [50] - 200 - minmax_a - Xa_u.csv')
    # 
    # %% [50, 25]
    print('sl = [50, 25]')
    sl = [50, 25]  # hidden layer structure
    runs = 200
    step = 0.1
    model = nn.NeuralNetwork(len(X_ori_train[0]), len(Y_train[0]), sl)
    print('### Start training ...')
    # Normal work much better than minmax
    
    # %% Norm He_u
    print('Norm He_u')
    model.train(X_ori_train, Y_train, runs, step, verbose=max(int(runs/20),1), 
                norm_method='normal', initializer='He_uniform')
    print('Training   accuracy:', model.accuracy_class(X_ori_train, Y_train))
    print('Evaluation accuracy:', model.accuracy_class(X_ori_eval, Y_eval))
    label_test = model.predict_class(feature_test)
    output(model.predict_class(feature_test), 
           'example_data/MNIST/test_predict - [50,25] - 200 - norm - He_u.csv')
    # 0.9652, 0.8913
    # %% Norm Xa_u
    print('Norm Xa_u')
    model.train(X_ori_train, Y_train, runs, step, verbose=max(int(runs/20),1), 
                norm_method='normal', initializer='Xavier_uniform')
    print('Training   accuracy:', model.accuracy_class(X_ori_train, Y_train))
    print('Evaluation accuracy:', model.accuracy_class(X_ori_eval, Y_eval))
    label_test = model.predict_class(feature_test)
    output(model.predict_class(feature_test), 
           'example_data/MNIST/test_predict - [50,25] - 200 - norm - Xa_u.csv')
    # 0.9285, 0.8487
    # %% Minmax_all He_u
    print('Minmax_all He_u')
    model.train(X_ori_train, Y_train, runs, step, verbose=max(int(runs/20),1), 
                norm_method='minmax_all', initializer='He_uniform')
    print('Training   accuracy:', model.accuracy_class(X_ori_train, Y_train))
    print('Evaluation accuracy:', model.accuracy_class(X_ori_eval, Y_eval))
    label_test = model.predict_class(feature_test)
    output(model.predict_class(feature_test), 
           'example_data/MNIST/test_predict - [50,25] - 200 - minmax_a - He_u.csv')
    # 0.11, 0.12
    # %% Minmax_all Xa_u
    print('Minmax_all Xa_u')
    model.train(X_ori_train, Y_train, runs, step, verbose=max(int(runs/20),1), 
                norm_method='minmax_all', initializer='Xavier_uniform')
    print('Training   accuracy:', model.accuracy_class(X_ori_train, Y_train))
    print('Evaluation accuracy:', model.accuracy_class(X_ori_eval, Y_eval))
    label_test = model.predict_class(feature_test)
    output(model.predict_class(feature_test), 
           'example_data/MNIST/test_predict - [50,25] - 200 - minmax_a - Xa_u.csv')
    # 0.11, 0.12
    # %% Minmax_all He_n
    print('Minmax_all He_n')
    model.train(X_ori_train, Y_train, runs, step, verbose=max(int(runs/20),1), 
                norm_method='minmax_all', initializer='He_normal')
    print('Training   accuracy:', model.accuracy_class(X_ori_train, Y_train))
    print('Evaluation accuracy:', model.accuracy_class(X_ori_eval, Y_eval))
    label_test = model.predict_class(feature_test)
    output(model.predict_class(feature_test), 
           'example_data/MNIST/test_predict - [50,25] - 200 - minmax_a - He_n.csv')
    # 0.11, 0.12
    # %% Minmax_all Xa_n
    print('Minmax_all Xa_n')
    model.train(X_ori_train, Y_train, runs, step, verbose=max(int(runs/20),1), 
                norm_method='minmax_all', initializer='Xavier_normal')
    print('Training   accuracy:', model.accuracy_class(X_ori_train, Y_train))
    print('Evaluation accuracy:', model.accuracy_class(X_ori_eval, Y_eval))
    label_test = model.predict_class(feature_test)
    output(model.predict_class(feature_test), 
           'example_data/MNIST/test_predict - [50,25] - 200 - minmax_a - Xa_n.csv')
    # 0.11, 0.12
    # %%
    ###### Evaluating the network ######
    
    ###### Output predicts for test data ######
    if output_predict:
        print('--- Output predicts for test data ---')
        '''
        if use_PCA:
            feature_test = pr.pca_transform(feature_test, pca_model, n_rf)
        '''
        label_test = model.predict_class(feature_test)
        examine(feature_test, label_test)
        output(label_test, 'example_data/MNIST/test_predict.csv')
