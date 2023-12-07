from euler import *

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband

import os
import pickle
from time import time, localtime, strftime

#GPU setup
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #change this to the id of the GPU you can use (e.g., "2")


def experiment_ESN_regression(units, input_scaling, bias_scaling, spectral_radius, leaky, input_to_readout, x_train,y_train,x_test,y_test):
        
    reservoir = keras.layers.RNN(cell = ReservoirCell(units = units, input_scaling = input_scaling, bias_scaling = bias_scaling, spectral_radius = spectral_radius,
                                                          leaky = leaky), return_sequences = True)
    readout = Ridge(solver = 'svd')


    #compute the reservoir states for the training sequences
    x_train_states = reservoir(x_train)
    if (input_to_readout):
        x_train_states = np.concatenate((x_train_states,x_train),axis = 2)
    #reshape the input for the linear regressor
    x_train_states_r = np.reshape(x_train_states,[-1,x_train_states.shape[2]])
    #train the linear readout
    readout.fit(x_train_states_r, y_train)
    #compute the output on the training set
    o_train = readout.predict(x_train_states_r)
    #compute the RMSE
    e_train = np.linalg.norm(o_train - y_train) / np.sqrt(len(y_train))
    
    #compute the reservoir states on the test set
    x_test_states = reservoir(x_test)
    if (input_to_readout):
        x_test_states = np.concatenate((x_test_states,x_test),axis = 2)

    #reshape the input for the linear regressor
    x_test_states_r = np.reshape(x_test_states,[-1,x_test_states.shape[2]])
    #compute the output on the test set
    o_test = readout.predict(x_test_states_r)
    #comput the RMSE on the test set
    e_test = np.linalg.norm(o_test - y_test) / np.sqrt(len(y_test))
    
    return (e_test, e_train)

def experiment_EuSN_regression(units, epsilon, gamma, input_scaling, bias_scaling, recurrent_scaling, input_to_readout, x_train,y_train,x_test,y_test):
        
    reservoir = keras.layers.RNN(cell = EulerReservoirCell(units = units, input_scaling = input_scaling, bias_scaling = bias_scaling, 
                                                           recurrent_scaling = recurrent_scaling,
                                                           epsilon = epsilon, gamma = gamma), return_sequences = True)
    readout = Ridge(solver = 'svd')


    #compute the reservoir states for the training sequences
    x_train_states = reservoir(x_train)
    if (input_to_readout):
        x_train_states = np.concatenate((x_train_states,x_train),axis = 2)
    #reshape the input for the linear regressor
    x_train_states_r = np.reshape(x_train_states,[-1,x_train_states.shape[2]])
    #train the linear readout
    readout.fit(x_train_states_r, y_train)
    #compute the output on the training set
    o_train = readout.predict(x_train_states_r)
    #compute the RMSE
    e_train = np.linalg.norm(o_train - y_train) / np.sqrt(len(y_train))
    
    #compute the reservoir states on the test set
    x_test_states = reservoir(x_test)
    if (input_to_readout):
        x_test_states = np.concatenate((x_test_states,x_test),axis = 2)

    #reshape the input for the linear regressor
    x_test_states_r = np.reshape(x_test_states,[-1,x_test_states.shape[2]])
    #compute the output on the test set
    o_test = readout.predict(x_test_states_r)
    #comput the RMSE on the test set
    e_test = np.linalg.norm(o_test - y_test) / np.sqrt(len(y_test))
    
    return (e_test, e_train)


#common experimental setting:
units = 200
max_trials = 100 
num_repetitions = 10 #10

#dataset-specific experimental setting:
dataset_names = ['Adiac',
                 'ECG5000',   
                 'HandOutlines',
                 'Mallat',
                 'ShapesAll',
                 'Trace',
                 'UWaveGestureLibraryAll',
                 'Wafer',
                 'Yoga',
                 'FordA',]
                 



root_path = './' #set this one to the root folder for the experiments
datasets_path = os.path.join(root_path,'datasets') #this subfolder contains the datasets


for dataset_name in dataset_names:
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print('** Starting experiments on dataset {} ** \n'.format(dataset_name))



    keras_dataset_filename = os.path.join(keras_datasets_path,dataset_name+'_dataset.p')
    dataset = pickle.load(open(keras_dataset_filename,"rb"))
    x_train_all,y_train_all,x_test, y_test,x_train, x_val, y_train, y_val = dataset[0],dataset[1],dataset[2],dataset[3],dataset[4],dataset[5],dataset[6],dataset[7]

    
    ## dataset organization ##
    
    #set the autoregressive task, where the target value at a certain time-step t
    #is given by the input at the following time set t+1

    #whole training set
    x_train_all_ = x_train_all[:,:-1,:]
    y_train_all_ = x_train_all[:,1:,:]
    #test set
    x_test_ = x_test[:,:-1,:]
    y_test_ = x_test[:,1:,:]
    #training set
    x_train_ = x_train[:,:-1,:]
    y_train_ = x_train[:,1:,:]
    #validation set
    x_val_ = x_val[:,:-1,:]
    y_val_ = x_val[:,1:,:]
    
    #reshaped target information for the different splits of the dataset
    y_test_r = np.reshape(y_test_,[-1,y_test_.shape[2]])
    y_train_r = np.reshape(y_train_,[-1,y_train_.shape[2]])
    y_val_r = np.reshape(y_val_,[-1,y_val_.shape[2]])
    y_train_all_r = np.reshape(y_train_all_,[-1,y_train_all_.shape[2]])


    

    results_path = os.path.join(root_path, 'results_ts_modeling',dataset_name)
    #create the results path if it does not exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)


    #experiments with the ESN model
    # --- ESN with input-readout connections ---------------------------------------------------------------------------------
    model_type = 'ESN_input'
    input_to_readout = True
       
    
    best_val_error = np.inf
    for i in range(max_trials):

        leaky = 10**(np.random.randint(-5,high = 1))
        input_scaling = 10**(np.random.randint(-3,high = 2))
        spectral_radius = 0.3*(np.random.randint(1,high = 5))
        bias_scaling = 10**(np.random.randint(-3,high = 2))

        (e_val, e_train) = experiment_ESN_regression(units = units, input_scaling = input_scaling, bias_scaling = bias_scaling, 
                                                 leaky = leaky, spectral_radius = spectral_radius,input_to_readout = input_to_readout,
                                                 x_train = x_train_,y_train = y_train_r, x_test = x_val_, y_test = y_val_r)

        if (e_val < best_val_error):
            best_val_error = e_val
            best_leaky = leaky
            best_input_scaling = input_scaling
            best_bias_scaling = bias_scaling
            best_spectral_radius = spectral_radius
            best_val_error = e_val
            print('{} - Best val error = {}; sr = {}; l = {}; in = {}; bs = {}'.format(
                i,best_val_error, spectral_radius, leaky, input_scaling, bias_scaling))

    print('Model selection completed')

    train_err = []
    test_err = []
    for i in range(num_repetitions):
        leaky = best_leaky
        input_scaling = best_input_scaling
        spectral_radius = best_spectral_radius
        bias_scaling = best_bias_scaling
        (e_test, e_train) = experiment_ESN_regression(units = units, input_scaling = input_scaling, bias_scaling = bias_scaling, 
                                                     leaky = leaky, spectral_radius = spectral_radius,input_to_readout = input_to_readout,
                                                     x_train = x_train_all_,y_train = y_train_all_r, x_test = x_test_, y_test = y_test_r)
        test_err.append(e_test)
        train_err.append(e_train)

    print('--'+model_type+' on {}--'.format(dataset_name))
    print('Results TS: MEAN {} STD {}'.format(np.mean(test_err),np.std(test_err)))
    print('Results TR: MEAN {} STD {}'.format(np.mean(train_err),np.std(train_err)))
    print('HP: spectral radius = {}; leaky = {}; input scaling = {}; bias scaling = {}'.format(
        best_spectral_radius, best_leaky, best_input_scaling, best_bias_scaling))
    
    
    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')

    results_logger = open(results_logger_filename,'w')
    results_logger.write('** Results:\n')
    results_logger.write('HP: spectral radius = {}; leaky = {}; input scaling = {}; bias scaling = {}\n'.format(
        best_spectral_radius, best_leaky, best_input_scaling, best_bias_scaling))
    results_logger.write('Results TS: MEAN {} STD {}\n'.format(np.mean(test_err),np.std(test_err)))
    results_logger.write('Results TR: MEAN {} STD {}\n'.format(np.mean(train_err),np.std(train_err)))
    results_logger.close()
    

    #experiments with the ESN model
    # --- ESN without input-readout connections ---------------------------------------------------------------------------------
    model_type = 'ESN_noinput'
    input_to_readout = False
       
    
    best_val_error = np.inf
    for i in range(max_trials):

        leaky = 10**(np.random.randint(-5,high = 1))
        input_scaling = 10**(np.random.randint(-3,high = 2))
        spectral_radius = 0.3*(np.random.randint(1,high = 5))
        bias_scaling = 10**(np.random.randint(-3,high = 2))

        (e_val, e_train) = experiment_ESN_regression(units = units, input_scaling = input_scaling, bias_scaling = bias_scaling, 
                                                 leaky = leaky, spectral_radius = spectral_radius,input_to_readout = input_to_readout,
                                                 x_train = x_train_,y_train = y_train_r, x_test = x_val_, y_test = y_val_r)

        if (e_val < best_val_error):
            best_val_error = e_val
            best_leaky = leaky
            best_input_scaling = input_scaling
            best_bias_scaling = bias_scaling
            best_spectral_radius = spectral_radius
            best_val_error = e_val
            print('{} - Best val error = {}; sr = {}; l = {}; in = {}; bs = {}'.format(
                i,best_val_error, spectral_radius, leaky, input_scaling, bias_scaling))

    print('Model selection completed')

    train_err = []
    test_err = []
    for i in range(num_repetitions):
        leaky = best_leaky
        input_scaling = best_input_scaling
        spectral_radius = best_spectral_radius
        bias_scaling = best_bias_scaling
        (e_test, e_train) = experiment_ESN_regression(units = units, input_scaling = input_scaling, bias_scaling = bias_scaling, 
                                                     leaky = leaky, spectral_radius = spectral_radius,input_to_readout = input_to_readout,
                                                     x_train = x_train_all_,y_train = y_train_all_r, x_test = x_test_, y_test = y_test_r)
        test_err.append(e_test)
        train_err.append(e_train)

    print('--'+model_type+' on {}--'.format(dataset_name))
    print('Results TS: MEAN {} STD {}'.format(np.mean(test_err),np.std(test_err)))
    print('Results TR: MEAN {} STD {}'.format(np.mean(train_err),np.std(train_err)))
    print('HP: spectral radius = {}; leaky = {}; input scaling = {}; bias scaling = {}'.format(
        best_spectral_radius, best_leaky, best_input_scaling, best_bias_scaling))
    
    
    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')

    results_logger = open(results_logger_filename,'w')
    results_logger.write('** Results:\n')
    results_logger.write('HP: spectral radius = {}; leaky = {}; input scaling = {}; bias scaling = {}\n'.format(
        best_spectral_radius, best_leaky, best_input_scaling, best_bias_scaling))
    results_logger.write('Results TS: MEAN {} STD {}\n'.format(np.mean(test_err),np.std(test_err)))
    results_logger.write('Results TR: MEAN {} STD {}\n'.format(np.mean(train_err),np.std(train_err)))
    results_logger.close()
        
        
    #experiments with the EuSN model
    # --- EuSN with input-readout connections ---------------------------------------------------------------------------------
    model_type = 'EuSN_input'
    input_to_readout = True
       
    
    best_val_error = np.inf
    for i in range(max_trials):

        gamma = 10**(np.random.randint(-5,high = 1))
        epsilon = 10**(np.random.randint(-5,high = 1))
        input_scaling = 10**(np.random.randint(-3,high = 2))
        recurrent_scaling = 10**(np.random.randint(-3,high = 2))
        bias_scaling = 10**(np.random.randint(-3,high = 2))

        (e_val, e_train) = experiment_EuSN_regression(units = units, input_scaling = input_scaling, bias_scaling = bias_scaling, 
                                                      recurrent_scaling = recurrent_scaling,epsilon = epsilon, gamma = gamma,
                                                      input_to_readout = input_to_readout,
                                                      x_train = x_train_,y_train = y_train_r, x_test = x_val_, y_test = y_val_r)

        if (e_val < best_val_error):
            print(e_val)
            best_val_error = e_val
            best_epsilon = epsilon
            best_gamma = gamma
            best_input_scaling = input_scaling
            best_bias_scaling = bias_scaling
            best_recurrent_scaling = recurrent_scaling
            best_val_error = e_val
            print('{} - Best val error = {}; e = {}; g = {}; rs = {}; in = {}; bs = {}'.format(
                i,best_val_error, epsilon, gamma, recurrent_scaling, input_scaling, bias_scaling))

    print('Model selection completed')

    train_err = []
    test_err = []
    for i in range(num_repetitions):
        epsilon = best_epsilon
        gamma = best_gamma
        input_scaling = best_input_scaling
        recurrent_scaling = best_recurrent_scaling
        bias_scaling = best_bias_scaling
        (e_test, e_train) = experiment_EuSN_regression(units = units, input_scaling = input_scaling, bias_scaling = bias_scaling, recurrent_scaling = best_recurrent_scaling,
                                                       epsilon = best_epsilon, gamma = best_gamma,input_to_readout = input_to_readout,
                                                       x_train = x_train_all_,y_train = y_train_all_r, x_test = x_test_, y_test = y_test_r)
        test_err.append(e_test)
        train_err.append(e_train)

    print('--'+model_type+' on {}--'.format(dataset_name))
    print('Results TS: MEAN {} STD {}'.format(np.mean(test_err),np.std(test_err)))
    print('Results TR: MEAN {} STD {}'.format(np.mean(train_err),np.std(train_err)))
    print('HP: epsilon = {}; gamma = {}; recurrent scaling = {}; input scaling = {}; bias scaling = {}'.format(
        best_epsilon, best_gamma, best_recurrent_scaling, best_input_scaling, best_bias_scaling))
    
    
    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')

    results_logger = open(results_logger_filename,'w')
    results_logger.write('** Results:\n')
    results_logger.write('HP: epsilon = {}; gamma = {}; recurrent scaling = {}; input scaling = {}; bias scaling = {}\n'.format(
        best_epsilon, best_gamma, best_recurrent_scaling, best_input_scaling, best_bias_scaling))
    results_logger.write('Results TS: MEAN {} STD {}\n'.format(np.mean(test_err),np.std(test_err)))
    results_logger.write('Results TR: MEAN {} STD {}\n'.format(np.mean(train_err),np.std(train_err)))
    results_logger.close()
    
    #experiments with the EuSN model
    # --- EuSN without input-readout connections ---------------------------------------------------------------------------------
    model_type = 'EuSN_noinput'
    input_to_readout = False
       
    
    best_val_error = np.inf
    for i in range(max_trials):

        gamma = 10**(np.random.randint(-5,high = 1))
        epsilon = 10**(np.random.randint(-5,high = 1))
        input_scaling = 10**(np.random.randint(-3,high = 2))
        recurrent_scaling = 10**(np.random.randint(-3,high = 2))
        bias_scaling = 10**(np.random.randint(-3,high = 2))

        (e_val, e_train) = experiment_EuSN_regression(units = units, input_scaling = input_scaling, bias_scaling = bias_scaling, 
                                                      recurrent_scaling = recurrent_scaling,epsilon = epsilon, gamma = gamma,
                                                      input_to_readout = input_to_readout,
                                                      x_train = x_train_,y_train = y_train_r, x_test = x_val_, y_test = y_val_r)

        if (e_val < best_val_error):
            print(e_val)
            best_val_error = e_val
            best_epsilon = epsilon
            best_gamma = gamma
            best_input_scaling = input_scaling
            best_bias_scaling = bias_scaling
            best_recurrent_scaling = recurrent_scaling
            best_val_error = e_val
            print('{} - Best val error = {}; e = {}; g = {}; rs = {}; in = {}; bs = {}'.format(
                i,best_val_error, epsilon, gamma, recurrent_scaling, input_scaling, bias_scaling))

    print('Model selection completed')

    train_err = []
    test_err = []
    for i in range(num_repetitions):
        epsilon = best_epsilon
        gamma = best_gamma
        input_scaling = best_input_scaling
        recurrent_scaling = best_recurrent_scaling
        bias_scaling = best_bias_scaling
        (e_test, e_train) = experiment_EuSN_regression(units = units, input_scaling = input_scaling, bias_scaling = bias_scaling, recurrent_scaling = best_recurrent_scaling,
                                                       epsilon = best_epsilon, gamma = best_gamma,input_to_readout = input_to_readout,
                                                       x_train = x_train_all_,y_train = y_train_all_r, x_test = x_test_, y_test = y_test_r)
        test_err.append(e_test)
        train_err.append(e_train)

    print('--'+model_type+' on {}--'.format(dataset_name))
    print('Results TS: MEAN {} STD {}'.format(np.mean(test_err),np.std(test_err)))
    print('Results TR: MEAN {} STD {}'.format(np.mean(train_err),np.std(train_err)))
    print('HP: epsilon = {}; gamma = {}; recurrent scaling = {}; input scaling = {}; bias scaling = {}'.format(
        best_epsilon, best_gamma, best_recurrent_scaling, best_input_scaling, best_bias_scaling))
    
    
    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')

    results_logger = open(results_logger_filename,'w')
    results_logger.write('** Results:\n')
    results_logger.write('HP: epsilon = {}; gamma = {}; recurrent scaling = {}; input scaling = {}; bias scaling = {}\n'.format(
        best_epsilon, best_gamma, best_recurrent_scaling, best_input_scaling, best_bias_scaling))
    results_logger.write('Results TS: MEAN {} STD {}\n'.format(np.mean(test_err),np.std(test_err)))
    results_logger.write('Results TR: MEAN {} STD {}\n'.format(np.mean(train_err),np.std(train_err)))
    results_logger.close()