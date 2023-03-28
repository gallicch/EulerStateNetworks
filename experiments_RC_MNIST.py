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


#common experimental setting:
num_guesses = 10 #number of reservoir guesses for final evaluation after model selection
max_trials = 200 #number of configurations to be sampled (randomly) during model selection
num_guesses_ms = 1 #number of guesses needed for model selection (in this case 1 is sufficient)
max_time = 60*60*10 #number of seconds that a model selection process is allowed to last = 10 h


#I used this version for the MNIST dataset in the experiments reported in the paper
#the only difference is that here we use the buffered versions of the ESN, RESN and EuSN classes
#to allow dealing with larger datasets, still using a one-shot training algorithm
dataset_names = ['MNIST']
                 



root_path = './' #set this one to the root folder for the experiments
datasets_path = os.path.join(root_path,'datasets') #this subfolder contains the datasets


for dataset_name in dataset_names:
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print('** Starting experiments on dataset {} ** \n'.format(dataset_name))


    #load the dataset
    keras_dataset_filename = os.path.join(datasets_path,dataset_name+'_dataset.p')
    dataset = pickle.load(open(keras_dataset_filename,"rb"))
    x_train_all,y_train_all,x_test, y_test,x_train, x_val, y_train, y_val = dataset[0],dataset[1],dataset[2],dataset[3],dataset[4],dataset[5],dataset[6],dataset[7]


    results_path = os.path.join(root_path, 'results',dataset_name)
    #create the results path if it does not exists
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ######
    #EuSN experiments 
    
    model_type = 'EuSN'
    
    model_selection_times_filename = os.path.join(results_path,'model_selection_times_'+model_type+'.p')
    times_filename = os.path.join(results_path,'times_'+model_type+'.p')
    accuracy_filename = os.path.join(results_path,'accuracy_'+model_type+'.p')
    tuner_filename = os.path.join(results_path,'tuner_'+model_type+'.p')

    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')

    results_logger = open(results_logger_filename,'w')
    results_logger.write('Experiment with '+model_type+' on dataset '+ dataset_name + ' starting now\n')
    time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('** local time = '+ time_string_start+'\n')

    initial_model_selection_time = time()

    # --- model selection start ---
    best_val_score = 0
    for i in range(max_trials):
        if (time()-initial_model_selection_time > max_time):
            print('--> terminating the model selection for exceeding max time after {} configurations.\n'.format(i))
            results_logger.write('--> terminating the model selection for exceeding max time after {} configurations.\n'.format(i))
            break
        
        
        num_units = 10 * np.random.randint(1,high = 21)
        gamma = 10**(np.random.randint(-5,high = 1))
        epsilon = 10**(np.random.randint(-5,high = 1))

        input_scaling = 10**(np.random.randint(-3,high = 2))
        recurrent_scaling = 10**(np.random.randint(-3,high = 2))
        bias_scaling = 10**(np.random.randint(-3,high = 2))
        
        val_score = 0
        for j in range(num_guesses_ms):

            model = EuSN_buffer(num_units, epsilon = epsilon, gamma = gamma, 
                          input_scaling = input_scaling, bias_scaling = bias_scaling,
                          recurrent_scaling= recurrent_scaling)
            model.fit(x_train,y_train)
            val_score = val_score + model.evaluate(x_val,y_val)
        val_score = val_score / num_guesses_ms
        
        if (val_score > best_val_score):
            best_gamma = gamma
            best_epsilon = epsilon
            best_num_units = num_units
            best_input_scaling = input_scaling
            best_bias_scaling = bias_scaling
            best_recurrent_scaling = recurrent_scaling
            best_val_score = val_score
            print('{} - Best Score = {}'.format(i,best_val_score))
    print('Model selection completed')        
    # --- model selection end ---
    elapsed_model_selection_time = time()-initial_model_selection_time
    time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('model selection concluded at local time = '+ time_string+'\n')


    #repeat the experiments with the best hyper-parameters. evaluate on the test set
    acc_ts = []
    required_time = []
    for i in range(num_guesses):
      initial_time = time()
      model = EuSN_buffer(best_num_units,epsilon = best_epsilon, gamma = best_gamma,input_scaling = best_input_scaling,
                     bias_scaling = best_bias_scaling,recurrent_scaling = best_recurrent_scaling)
      model.fit(x_train, y_train)
      acc = model.evaluate(x_test,y_test)
      required_time.append(time()-initial_time)
      acc_ts.append(acc)

    time_string_end = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('*** best model assessment concluded at local time = '+ time_string_end+'\n')


    with open(model_selection_times_filename, 'wb') as f:
        pickle.dump(elapsed_model_selection_time, f)
    with open(times_filename, 'wb') as f:
        pickle.dump(required_time, f)
    with open(accuracy_filename, 'wb') as f:
        pickle.dump(acc_ts, f)    


    print('--'+model_type+' on {}--'.format(dataset_name))
    print('Results: MEAN {} STD {}'.format(np.mean(acc_ts),np.std(acc_ts)))
    print('----- required time: MEAN {} STD {}'.format(np.mean(required_time),np.std(required_time)))
    print('----- total model selection time: {}'.format(elapsed_model_selection_time))



    results_logger.write('** Results:\n')
    results_logger.write('Start time: '+time_string_start+'\n')
    results_logger.write('End time: '+time_string_end+'\n')
    results_logger.write('Accuracy: MEAN {} STD {}\n'.format(np.mean(acc_ts),np.std(acc_ts)))
    results_logger.write('Model selection time: {} seconds = {} minutes\n'.format(elapsed_model_selection_time, elapsed_model_selection_time/60))
    results_logger.write('Average time for TR,TS: MEAN {} STD {}\n'.format(np.mean(required_time),np.std(required_time)))
    results_logger.write('Model summary:\n')
    results_logger.write('trainable parameters = {}\n'.format(model.readout.coef_.shape[0] * model.readout.coef_.shape[1] + model.readout.intercept_.shape[0]))
    results_logger.write('reservoir size = {}; epsilon = {}; gamma = {}\n'.format(best_num_units, best_epsilon, best_gamma))
    results_logger.write('input scaling = {}; bias scaling = {}; recurrent scaling = {}\n'.format(best_input_scaling, best_bias_scaling, best_recurrent_scaling))
    results_logger.close()


    #######
    # ESN experiments
    model_type = 'ESN'

    model_selection_times_filename = os.path.join(results_path,'model_selection_times_'+model_type+'.p')
    times_filename = os.path.join(results_path,'times_'+model_type+'.p')
    accuracy_filename = os.path.join(results_path,'accuracy_'+model_type+'.p')

    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')

    results_logger = open(results_logger_filename,'w')
    results_logger.write('Experiment with '+model_type+' on dataset '+ dataset_name + ' starting now\n')
    time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('** local time = '+ time_string_start+'\n')

    initial_model_selection_time = time()

    # --- model selection start ---
    best_val_score = 0
    for i in range(max_trials):
        if (time()-initial_model_selection_time > max_time):
            print('--> terminating the model selection for exceeding max time after {} configurations.\n'.format(i))
            results_logger.write('--> terminating the model selection for exceeding max time after {} configurations.\n'.format(i))
            break
        
        
        num_units = 10 * np.random.randint(1,high = 21)
        leaky = 10**(np.random.randint(-5,high = 1))

        input_scaling = 10**(np.random.randint(-3,high = 2))
        spectral_radius = 0.1*(np.random.randint(1,high = 16))
        bias_scaling = 10**(np.random.randint(-3,high = 2))
        
        val_score = 0
        for j in range(num_guesses_ms):
            model = ESN_buffer(num_units, leaky = leaky,
                      input_scaling = input_scaling, bias_scaling = bias_scaling,
                      spectral_radius= spectral_radius)
            model.fit(x_train,y_train)
            val_score = val_score + model.evaluate(x_val,y_val)
        val_score = val_score / num_guesses_ms

        if (val_score > best_val_score):
            best_leaky = leaky
            best_num_units = num_units
            best_input_scaling = input_scaling
            best_bias_scaling = bias_scaling
            best_spectral_radius = spectral_radius
            best_val_score = val_score
            print('{} - Best Score = {}'.format(i,best_val_score))
    print('Model selection completed')        
    # --- model selection end ---
    elapsed_model_selection_time = time()-initial_model_selection_time
    time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('model selection concluded at local time = '+ time_string+'\n')



    #repeat the experiments with the best hyper-parameters. evaluate on the test set
    acc_ts = []
    required_time = []
    for i in range(num_guesses):
      initial_time = time()
      model = ESN_buffer(best_num_units,leaky = best_leaky,input_scaling = best_input_scaling,
                     bias_scaling = best_bias_scaling,spectral_radius = best_spectral_radius)
      model.fit(x_train, y_train)
      acc = model.evaluate(x_test,y_test)
      required_time.append(time()-initial_time)
      acc_ts.append(acc)

    time_string_end = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('*** best model assessment concluded at local time = '+ time_string_end+'\n')


    with open(model_selection_times_filename, 'wb') as f:
        pickle.dump(elapsed_model_selection_time, f)
    with open(times_filename, 'wb') as f:
        pickle.dump(required_time, f)
    with open(accuracy_filename, 'wb') as f:
        pickle.dump(acc_ts, f)      

    print('--'+model_type+' on {}--'.format(dataset_name))
    print('Results: MEAN {} STD {}'.format(np.mean(acc_ts),np.std(acc_ts)))
    print('----- required time: MEAN {} STD {}'.format(np.mean(required_time),np.std(required_time)))
    print('----- total model selection time: {}'.format(elapsed_model_selection_time))



    results_logger.write('** Results:\n')
    results_logger.write('Start time: '+time_string_start+'\n')
    results_logger.write('End time: '+time_string_end+'\n')
    results_logger.write('Accuracy: MEAN {} STD {}\n'.format(np.mean(acc_ts),np.std(acc_ts)))
    results_logger.write('Model selection time: {} seconds = {} minutes\n'.format(elapsed_model_selection_time, elapsed_model_selection_time/60))
    results_logger.write('Average time for TR,TS: MEAN {} STD {}\n'.format(np.mean(required_time),np.std(required_time)))
    results_logger.write('Model summary:\n')
    results_logger.write('trainable parameters = {}\n'.format(model.readout.coef_.shape[0] * model.readout.coef_.shape[1] + model.readout.intercept_.shape[0]))
    results_logger.write('reservoir size = {}; leaky = {}\n'.format(best_num_units, best_leaky))
    results_logger.write('input scaling = {}; bias scaling = {}; spectral radius = {}\n'.format(best_input_scaling, best_bias_scaling, best_spectral_radius))
    results_logger.close()


    #######
    # R-ESN experiments
    model_type = 'RESN'

    model_selection_times_filename = os.path.join(results_path,'model_selection_times_'+model_type+'.p')
    times_filename = os.path.join(results_path,'times_'+model_type+'.p')
    accuracy_filename = os.path.join(results_path,'accuracy_'+model_type+'.p')

    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')

    results_logger = open(results_logger_filename,'w')
    results_logger.write('Experiment with '+model_type+' on dataset '+ dataset_name + ' starting now\n')
    time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('** local time = '+ time_string_start+'\n')

    initial_model_selection_time = time()

    # --- model selection start ---
    best_val_score = 0
    for i in range(max_trials):
        if (time()-initial_model_selection_time > max_time):
            print('--> terminating the model selection for exceeding max time after {} configurations.\n'.format(i))
            results_logger.write('--> terminating the model selection for exceeding max time after {} configurations.\n'.format(i))
            break
        
        num_units = 10 * np.random.randint(1,high = 21)
        leaky = 10**(np.random.randint(-5,high = 1))

        input_scaling = 10**(np.random.randint(-3,high = 2))
        spectral_radius = 0.1*(np.random.randint(1,high = 16))
        bias_scaling = 10**(np.random.randint(-3,high = 2))

        
        val_score = 0
        for j in range(num_guesses_ms):
            model = RESN_buffer(num_units, leaky = leaky,
                      input_scaling = input_scaling, bias_scaling = bias_scaling,
                      spectral_radius= spectral_radius)
            model.fit(x_train,y_train)
            val_score = val_score + model.evaluate(x_val,y_val)
        val_score = val_score / num_guesses_ms
        
        if (val_score > best_val_score):
            best_leaky = leaky
            best_num_units = num_units
            best_input_scaling = input_scaling
            best_bias_scaling = bias_scaling
            best_spectral_radius = spectral_radius
            best_val_score = val_score
            print('{} - Best Score = {}'.format(i,best_val_score))
    print('Model selection completed')        
    # --- model selection end ---
    elapsed_model_selection_time = time()-initial_model_selection_time
    time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('model selection concluded at local time = '+ time_string+'\n')


    #repeat the experiments with the best hyper-parameters. evaluate on the test set
    acc_ts = []
    required_time = []
    for i in range(num_guesses):
      initial_time = time()
      model = RESN_buffer(best_num_units,leaky = best_leaky,input_scaling = best_input_scaling,
                     bias_scaling = best_bias_scaling,spectral_radius =best_spectral_radius)
      model.fit(x_train, y_train)
      acc = model.evaluate(x_test,y_test)
      required_time.append(time()-initial_time)
      acc_ts.append(acc)

    time_string_end = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('*** best model assessment concluded at local time = '+ time_string_end+'\n')


    with open(model_selection_times_filename, 'wb') as f:
        pickle.dump(elapsed_model_selection_time, f)
    with open(times_filename, 'wb') as f:
        pickle.dump(required_time, f)
    with open(accuracy_filename, 'wb') as f:
        pickle.dump(acc_ts, f)      

    print('--'+model_type+' on {}--'.format(dataset_name))
    print('Results: MEAN {} STD {}'.format(np.mean(acc_ts),np.std(acc_ts)))
    print('----- required time: MEAN {} STD {}'.format(np.mean(required_time),np.std(required_time)))
    print('----- total model selection time: {}'.format(elapsed_model_selection_time))



    results_logger.write('** Results:\n')
    results_logger.write('Start time: '+time_string_start+'\n')
    results_logger.write('End time: '+time_string_end+'\n')
    results_logger.write('Accuracy: MEAN {} STD {}\n'.format(np.mean(acc_ts),np.std(acc_ts)))
    results_logger.write('Model selection time: {} seconds = {} minutes\n'.format(elapsed_model_selection_time, elapsed_model_selection_time/60))
    results_logger.write('Average time for TR,TS: MEAN {} STD {}\n'.format(np.mean(required_time),np.std(required_time)))
    results_logger.write('Model summary:\n')
    results_logger.write('trainable parameters = {}\n'.format(model.readout.coef_.shape[0] * model.readout.coef_.shape[1] + model.readout.intercept_.shape[0]))
    results_logger.write('reservoir size = {}; leaky = {}\n'.format(best_num_units, best_leaky))
    results_logger.write('input scaling = {}; bias scaling = {}; spectral radius = {}\n'.format(best_input_scaling, best_bias_scaling, best_spectral_radius))
    results_logger.close()

