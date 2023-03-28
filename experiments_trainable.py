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


#dataset-specific experimental setting:
dataset_names = ['Adiac',
                 'Blink',
                 'CharacterTrajectories',
                 'ECG5000',
                 'Epilepsy',
                 'FordA',
                 'HandOutlines',
                 'Heartbeat',
                 'IMDB_embedded',
                 'Libras',
                 'Mallat',
                 'MNIST',
                 'MotionSenseHAR',
                 'Reuters_embedded',
                 'ShapesAll',
                 'SpokenArabicDigits',
                 'Trace',
                 'UWaveGestureLibraryAll',
                 'Wafer',
                 'Yoga']
                 



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

    output_units = max(y_train_all)+1
    if (output_units==2):
        output_units = 1
    if (output_units==1):
        output_activation = 'sigmoid'
        loss_function = 'binary_crossentropy'
    else:
        output_activation = 'sigmoid'
        loss_function = 'sparse_categorical_crossentropy'

    results_path = os.path.join(root_path, 'results',dataset_name)
    #create the results path if it does not exists
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        

    ######
    #GRU experiments
    model_type = 'GRU'


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
        lr = 10**(np.random.randint(-5,high = 0))
        patience = 50 
        batch_size = 2**np.random.randint(5,high = 9)

        val_score = 0
        for j in range(num_guesses_ms):
            model = keras.Sequential([
                keras.layers.GRU(units = num_units),
                keras.layers.Dense(output_units, activation = output_activation)
            ])
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate = lr),loss=loss_function,metrics=['accuracy'])


            model.fit(x_train,y_train,
                     verbose = 0, 
                     epochs = num_epochs,
                     validation_data = (x_val,y_val),
                     batch_size = batch_size,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)])

            _,val_score_new = model.evaluate(x_val,y_val, verbose = 0)
            val_score = val_score + val_score_new
        val_score = val_score / num_guesses_ms
            
        print('RUN {}. Score = {}'.format(i,val_score))
        if (val_score > best_val_score):
            best_num_units = num_units
            best_lr = lr
            best_patience = patience
            best_batch_size = batch_size
            best_val_score = val_score
            print('{} - Best Score = {}'.format(i,best_val_score))
    print('Model selection completed')        
    # --- model selection end ---
    elapsed_model_selection_time = time()-initial_model_selection_time
    time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('model selection concluded at local time = '+ time_string+'\n')


    #choose the best hyper-parameters
    acc_ts = []
    required_time = []
    for i in range(num_guesses):
      initial_time = time()
      model = keras.Sequential([
        keras.layers.GRU(units = best_num_units),
        keras.layers.Dense(output_units, activation = output_activation)
      ])
      model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = best_lr),loss=loss_function,metrics=['accuracy'])

      model.fit(x_train,y_train,
             verbose = 0,
             epochs = num_epochs,
             validation_data = (x_val,y_val),
             batch_size = best_batch_size,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = best_patience)])
      _, acc = model.evaluate(x_test,y_test, verbose = 0)
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
    model.summary()


    results_logger.write('** Results:\n')
    results_logger.write('Start time: '+time_string_start+'\n')
    results_logger.write('End time: '+time_string_end+'\n')
    results_logger.write('Accuracy: MEAN {} STD {}\n'.format(np.mean(acc_ts),np.std(acc_ts)))
    results_logger.write('Model selection time: {} seconds = {} minutes\n'.format(elapsed_model_selection_time, elapsed_model_selection_time/60))
    results_logger.write('Average time for TR,TS: MEAN {} STD {}\n'.format(np.mean(required_time),np.std(required_time)))
    results_logger.write('Model summary:\n')
    results_logger.write('num units = {}; lr = {}; batch size = {}; patience = {} \n'.format(best_num_units, best_lr, best_batch_size, best_patience))
    model.summary(print_fn=lambda x: results_logger.write(x + '\n'))
    results_logger.close()


    ######
    #Antisymmetric RNN experiments
    model_type = 'AntisymmetricRNN'


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
        lr = 10**(np.random.randint(-5,high = 0))
        patience = 50
        batch_size = 2**np.random.randint(5,high = 9)
        gamma = 10**(np.random.randint(-5,high = 1))
        epsilon = 10**(np.random.randint(-5,high = 1))

        val_score = 0
        for j in range(num_guesses_ms):
            model = keras.Sequential([
                keras.layers.RNN(cell = AntisymmetricRNNCell(
                    units = num_units,
                    gamma = gamma,
                    epsilon = epsilon)),
                keras.layers.Dense(output_units, activation = output_activation)
            ])
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate = lr),loss=loss_function,metrics=['accuracy'])


            model.fit(x_train,y_train,
                     verbose = 0, 
                     epochs = num_epochs,
                     validation_data = (x_val,y_val),
                     batch_size = batch_size,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)])

            _,val_score_new = model.evaluate(x_val,y_val, verbose = 0)
            val_score = val_score + val_score_new
        val_score = val_score / num_guesses_ms
            
            
        print('RUN {}. Score = {}'.format(i,val_score))
        if (val_score > best_val_score):
            best_num_units = num_units
            best_epsilon = epsilon
            best_gamma = gamma
            best_lr = lr
            best_patience = patience
            best_batch_size = batch_size
            best_val_score = val_score
            print('{} - Best Score = {}'.format(i,best_val_score))
    print('Model selection completed')        
    # --- model selection end ---
    elapsed_model_selection_time = time()-initial_model_selection_time
    time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('model selection concluded at local time = '+ time_string+'\n')


    #choose the best hyper-parameters
    acc_ts = []
    required_time = []
    for i in range(num_guesses):
      initial_time = time()
      model = keras.Sequential([
        keras.layers.RNN(cell = AntisymmetricRNNCell(
            units = best_num_units,
            gamma = best_gamma,
            epsilon = best_epsilon)),
        keras.layers.Dense(output_units, activation = output_activation)
      ])
      model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = best_lr),loss=loss_function,metrics=['accuracy'])

      model.fit(x_train,y_train,
             verbose = 0,
             epochs = num_epochs,
             validation_data = (x_val,y_val),
             batch_size = best_batch_size,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = best_patience)])
      _, acc = model.evaluate(x_test,y_test, verbose = 0)
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
    model.summary()


    results_logger.write('** Results:\n')
    results_logger.write('Start time: '+time_string_start+'\n')
    results_logger.write('End time: '+time_string_end+'\n')
    results_logger.write('Accuracy: MEAN {} STD {}\n'.format(np.mean(acc_ts),np.std(acc_ts)))
    results_logger.write('Model selection time: {} seconds = {} minutes\n'.format(elapsed_model_selection_time, elapsed_model_selection_time/60))
    results_logger.write('Average time for TR,TS: MEAN {} STD {}\n'.format(np.mean(required_time),np.std(required_time)))
    results_logger.write('Model summary:\n')
    results_logger.write('num units = {}; lr = {}; batch size = {}; patience = {} \n'.format(best_num_units, best_lr, best_batch_size, best_patience))
    model.summary(print_fn=lambda x: results_logger.write(x + '\n'))
    results_logger.close()

    ######
    # vanilla RNN experiments
    model_type = 'SimpleRNN'

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
        lr = 10**(np.random.randint(-5,high = 0))
        patience = 50
        batch_size = 2**np.random.randint(5,high = 9)

        val_score = 0
        for j in range(num_guesses_ms):
            model = keras.Sequential([
                keras.layers.SimpleRNN(units = num_units),
                keras.layers.Dense(output_units, activation = output_activation)
            ])
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate = lr),loss=loss_function,metrics=['accuracy'])


            model.fit(x_train,y_train,
                     verbose = 0, 
                     epochs = num_epochs,
                     validation_data = (x_val,y_val),
                     batch_size = batch_size,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)])

            _,val_score_new = model.evaluate(x_val,y_val, verbose = 0)
            val_score = val_score + val_score_new
        val_score = val_score / num_guesses_ms
            
            
        print('RUN {}. Score = {}'.format(i,val_score))
        if (val_score > best_val_score):
            best_num_units = num_units
            best_lr = lr
            best_patience = patience
            best_batch_size = batch_size
            best_val_score = val_score
            print('{} - Best Score = {}'.format(i,best_val_score))
    print('Model selection completed')        
    # --- model selection end ---
    elapsed_model_selection_time = time()-initial_model_selection_time
    time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('model selection concluded at local time = '+ time_string+'\n')


    #choose the best hyper-parameters
    acc_ts = []
    required_time = []
    for i in range(num_guesses):
      initial_time = time()
      model = keras.Sequential([
        keras.layers.SimpleRNN(units = best_num_units),
        keras.layers.Dense(output_units, activation = output_activation)
      ])
      model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = best_lr),loss=loss_function,metrics=['accuracy'])

      model.fit(x_train,y_train,
             verbose = 0,
             epochs = num_epochs,
             validation_data = (x_val,y_val),
             batch_size = best_batch_size,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = best_patience)])

      _, acc = model.evaluate(x_test,y_test, verbose = 0)
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
    model.summary()


    results_logger.write('** Results:\n')
    results_logger.write('Start time: '+time_string_start+'\n')
    results_logger.write('End time: '+time_string_end+'\n')
    results_logger.write('Accuracy: MEAN {} STD {}\n'.format(np.mean(acc_ts),np.std(acc_ts)))
    results_logger.write('Model selection time: {} seconds = {} minutes\n'.format(elapsed_model_selection_time, elapsed_model_selection_time/60))
    results_logger.write('Average time for TR,TS: MEAN {} STD {}\n'.format(np.mean(required_time),np.std(required_time)))
    results_logger.write('Model summary:\n')
    results_logger.write('num units = {}; lr = {}; batch size = {}; patience = {} \n'.format(best_num_units, best_lr, best_batch_size, best_patience))
    model.summary(print_fn=lambda x: results_logger.write(x + '\n'))
    results_logger.close()