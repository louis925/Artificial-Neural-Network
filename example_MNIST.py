import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn

length_x = 28 # width of the image
length_y = 28 # height of the image
#rescale_base = 255
test_split = 3000
print_wrong_cases = True
output_predict = True
test_layers_combination = True
use_PCA = False
tune_parameters = False
avg_run = 3

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
        print('i =',i, 'label =', label[i])
        display(feature[i])
    
# %%
def par_tune(feature_train, label_train):
#    for i in range(avg_run):
#        alpha= 0.01 * 10**(-1*i)
#        print("alpha= %f" % alpha)
    best_accuracy = -1.0
    
    if test_layers_combination:
        test_layers = (
            (50,), (300,), (500,), (700,),
            #(400), (600), (700), (800), (1200), (1600),
            #(400, 100), (800, 200), (1200, 300), (1600, 400),
            #(400, 100, 40), (800, 400, 200), (1600, 800, 200),
            #(400, 300, 200, 100), (400, 400, 400, 400), (800, 800, 800, 800)
        )
    else:
        test_layers = ((700,),)
    
    for tlayer in test_layers:
        print("********** Layers:", tlayer, "**********")
        n_test_tot = 0
        n_test_cor = 0
        for i in range(avg_run):
            train_f_shuff, train_l_shuff, test_f_shuff, test_l_shuff= make_train_test(feature_train, label_train, test_split)
            
            trained_nn = nn.train_neural_network(train_f_shuff, train_l_shuff, rescale_base, hidden_layer_sizes = tlayer, tol = 2e-5)
            n_cor_i, n_tot_i, wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, test_f_shuff, test_l_shuff, rescale_base)

            n_test_cor += n_cor_i
            n_test_tot += n_tot_i

#        print("***** PCA *****")
#        trained_nn = nn.train_neural_network(reduced_features[:n_train], label_train[:n_train], rescale_base)
#        wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, reduced_features[n_train:n_train_data], label_train[n_train:n_train_data], rescale_base)
        if best_accuracy < n_test_cor/n_test_tot:
            best_accuracy = n_test_cor/n_test_tot
            best_trained_nn = trained_nn
            best_trained_nn.name = tlayer
        print('--- Average accuracy: ', n_test_cor/n_test_tot, ' ---')
        print()

    
#    for i in range(10):
#        input_tol= 10**(-1*i)
#        print("Tol:", input_tol)
#        trained_nn = nn.train_neural_network(feature_train[0:n_data-test_split], label_train[0:n_data-test_split], rescale_base, hidden_layer_sizes = (700), tol= input_tol)
#        wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, feature_train[n_train:n_train_data], label_train[n_train:n_train_data], rescale_base)
#        print()

    print('Best neural network is', best_trained_nn.name)
    return best_trained_nn

def make_train_test(features, labels, n_test):
    '''
    Shuffle the training data and slpit them into training set and test set.
    '''
    n_data= len(labels)
#    combine= np.concatenate( (features, labels.reshape( (n_data, 1) )), axis=1 )
#    np.random.shuffle(combine)
#    return combine[0:(n_data-n_test), 0:-1], combine[0:(n_data-n_test), -1], combine[(n_data-n_test):, 0:-1], combine[(n_data-n_test):, -1]
    shuffle_index = list(range(n_data))
    np.random.shuffle(shuffle_index)
    features_shuff = np.array([features[i] for i in shuffle_index])
    labels_shuff = np.array([labels[i] for i in shuffle_index])
    return features_shuff[n_test:], labels_shuff[n_test:], features_shuff[:n_test], labels_shuff[:n_test]

#def main():
if __name__ == '__main__':
    print('=== Test on MNIST ===')

    ###### Reading data ######
    print('Reading data')
    (feature_train, label_train, feature_test) = read()

    ###### Print shapes ######
    print(feature_train.shape)
    print(label_train.shape)
    #    print(feature_train[1,])
    #    print(label_train[1])
    print(feature_test.shape)
    #pd.display(feature_train[1], length_x, length_y)
    
    n_class = 10  # number of classes
    
    X_ori_train = feature_train
    Y_train = np.array([[1 if label == i else 0 for i in range(n_class)] for label in label_train])
    
    sl = [200]
    
    runs = 100
    step = 0.1
    model = nn.NeuralNetwork(len(X_ori_train[0]), len(Y_train[0]), sl)
    model.train(X_ori_train, Y_train, runs, step, verbose=1, 
                norm_method='minmax')
    
    print('Training accuracy:', model.accuracy_class(X_ori_train, Y_train))
    
    '''
    ###### Perform PCA reduction ######
    if use_PCA:
        feature_train, pca_model, n_rf = pr.pca_reduction(feature_train)
        print(feature_train.shape)
    ###### Tune pars of NN ######
    if tune_parameters:
        trained_nn = par_tune(feature_train, label_train)        
    else:
        ## Training NN and get results ######
        train_f_shuff, train_l_shuff, test_f_shuff, test_l_shuff= make_train_test(feature_train, label_train, test_split)
            
        trained_nn = nn.train_neural_network(train_f_shuff, train_l_shuff, rescale_base, hidden_layer_sizes = (700,), tol= 2e-5, early_stopping = True)
        n_cor_i, n_tot_i, wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, test_f_shuff, test_l_shuff, rescale_base)        

    '''
    '''
    ###### Print out wrong cases ######
    if print_wrong_cases and not tune_parameters and not use_PCA:
        print('Wrong cases:')
        for i in range(min(10, len(wrong_l))):
            print("(i, label, predict)", i, ",", wrong_l[i], ",", wrong_p[i])
            display(wrong_f[i], length_x, length_y)
    '''
    
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
