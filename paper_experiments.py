"""
Code for replicating the experiments on the time-series classification
datasets reported in the paper

@author: gallicch
"""

import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband

from euler import *

import os
import pickle
from time import time, localtime, strftime

import gdown

#GPU setup
#use the following line to isolate a specif GPU in case you desire to run
#the experiments in a single-GPU mode (as in the paper)
#os.environ["CUDA_VISIBLE_DEVICES"]="3"

#common experimental setting:
num_epochs = 200
num_guesses = 10
patience = 10
max_units = 200
batch_size = 32

#dataset-specific experimental setting
#these are customized based on the specific task
output_units = None
output_activation = None
loss_function = None
dataset_name = None
root_path = './' #change this to the path where you want all your experiments data
keras_datasets_path = os.path.join(root_path,'datasets')
model_selection_path = os.path.join(root_path,'model_selection_data')

keras_dataset_filename = None #this one needs to be setup for the specific task 
results_path = None #this one needs to be setup for the specific task
  

#number of output units for the different tasks
task_output_units = {
    'Adiac': 37,
    'CharacterTrajectories':20,
    'ECG5000':5,
    'Epilepsy':4,
    'Heartbeat':1,
    'Libras':15,
    'ShapesAll':60,
    'Wafer':1,
    'HandOutlines':1,
    'IMDB_embedded':1,
    'Reuters_embedded':46,
    'SpokenArabicDigits':10
}

#location of the dataset files for download
task_filenames = {
    'Adiac':'https://drive.google.com/uc?id=1F6kajRTJg9o-nNr9De5q4pZaeyx_bIFm',
    'CharacterTrajectories':'https://drive.google.com/uc?id=1A0sXA8RxJS0wg3mlXuZJvlZlCpTq5ihr',
    'ECG5000':'https://drive.google.com/uc?id=1yOMn3-Phqqf53fPFF2X-huh3ilbUb9SB',
    'Epilepsy':'https://drive.google.com/uc?id=1CsfcIXeh1B95DOr9bcvtaa8SP7Ncx8oT',
    'Heartbeat':'https://drive.google.com/uc?id=1ESHU06RHM1p7H9JW6hoKSSSRCy8AfcAg',
    'Libras':'https://drive.google.com/uc?id=14EtYLNt9K2Dd4SX2V6la7O8RwySrzlUf',
    'ShapesAll':'https://drive.google.com/uc?id=1UvGPbl3YyFAEL824xmDsBjVgqYMLtGp7',
    'Wafer':'https://drive.google.com/uc?id=1UvGPbl3YyFAEL824xmDsBjVgqYMLtGp7',
    'HandOutlines':'https://drive.google.com/uc?id=1POvIhl3vuZRS3a_v9-v7D8ibVXdyKZnv',
    'IMDB_embedded':'https://drive.google.com/uc?id=1oLPQKcB0Gpu4iqpw1XM72v8sba4sYgvQ',
    'Reuters_embedded':'https://drive.google.com/uc?id=1NPCBzWQ605vMfcy5fv4wkGm_7FScoNnT',
    'SpokenArabicDigits':'https://drive.google.com/uc?id=19K1sMn53ZxBIabQ_bzpT67uZwOyxZ5Xr'
}




def download_dataset(dataset_name, destination_folder):
    #downloads the specified dataset in the given destination folder
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    url = task_filenames[dataset_name]
    output_filename = dataset_name + '_dataset.p'
    output_path = os.path.join(destination_folder, output_filename)
    gdown.download(url, output_path, quiet=False)
    #gdown.download(url, destination_folder, quiet=False)
    
    
def task_settings(task_name):
    global output_activation, loss_function, output_units
    if (task_name in task_output_units.keys()):
        output_units = task_output_units[task_name]
        if (output_units == 1):
            #in this case it is a binary classification task
            output_activation = 'sigmoid'
            loss_function = 'binary_crossentropy'
        else:
            #in this case it is a multiple class classification task
            output_activation = 'softmax'
            loss_function = 'sparse_categorical_crossentropy'
    else:
        print('Warning: the specified task name is not valid')

def load_task_data(task_name):
    global keras_datasets_path, keras_dataset_filename, root_path, results_path
    #create the folder where all the datasets go
    #if it does not exist already
    if not os.path.exists(keras_datasets_path):
        os.makedirs(keras_datasets_path)
    #download the dataset for the specific task
    #if it does not exist already
    keras_dataset_filename = os.path.join(keras_datasets_path,task_name+'_dataset.p')
    if not os.path.exists(keras_dataset_filename):
        download_dataset(task_name,keras_datasets_path)
    
    #load the data
    print(keras_dataset_filename)
    dataset = pickle.load(open(keras_dataset_filename,"rb"))
    x_train_all,y_train_all,x_test, y_test,x_train, x_val, y_train, y_val = dataset[0],dataset[1],dataset[2],dataset[3],dataset[4],dataset[5],dataset[6],dataset[7]

    results_path = os.path.join(root_path, 'results',task_name)
    #create the results path if it does not exists
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    return x_train_all,y_train_all,x_test, y_test,x_train, x_val, y_train, y_val




def build_model_EuSN(hp):
    #Euler State Network architecture
    model = EuSN(units = hp.Int('units', min_value = 5, max_value = max_units),
                output_units = output_units, 
                output_activation = output_activation,
                input_scaling = hp.Float('input_scaling', min_value = 0.01, max_value = 1.5, sampling = 'linear'),
                bias_scaling = hp.Float('bias_scaling', min_value = 0.01, max_value = 1.5, sampling = 'linear'),
                recurrent_scaling = hp.Float('recurrent_scaling', min_value = 0.01, max_value = 1.5, sampling = 'linear'),
                gamma = hp.Float('gamma', min_value = 0.00001, max_value = 0.1, sampling = 'log'),
                epsilon = hp.Float('epsilon', min_value = 0.00001, max_value = 0.1, sampling = 'log'))
    model.readout.compile(
        optimizer=keras.optimizers.RMSprop(
            hp.Float('learning_rate',min_value = 1e-5,max_value = 1e-1, sampling = 'log')),
        loss=loss_function,
        metrics=['accuracy'])
    
    return model

def build_model_ESN(hp):
    #Echo State Network architecture
    model = ESN(units = hp.Int('units', min_value = 5, max_value = max_units),
                output_units = output_units, 
                output_activation = output_activation,
                input_scaling = hp.Float('input_scaling', min_value = 0.01, max_value = 1.5, sampling = 'linear'),
                bias_scaling = hp.Float('bias_scaling', min_value = 0.01, max_value = 1.5, sampling = 'linear'),
                spectral_radius = hp.Float('spectral_radius', min_value = 0.01, max_value = 1.5, sampling = 'linear'),
                leaky = hp.Float('leaky', min_value = 0.01, max_value = 1, sampling = 'linear'))
    model.readout.compile(
        optimizer=keras.optimizers.RMSprop(
            hp.Float('learning_rate',min_value = 1e-5,max_value = 1e-1, sampling = 'log')),
        loss=loss_function,
        metrics=['accuracy'])
    
    return model


def build_model_RESN(hp):
    #Echo State Network with ring reservoir
    model = RESN(units = hp.Int('units', min_value = 5, max_value = max_units),
                output_units = output_units, 
                output_activation = output_activation,
                input_scaling = hp.Float('input_scaling', min_value = 0.01, max_value = 1.5, sampling = 'linear'),
                bias_scaling = hp.Float('bias_scaling', min_value = 0.01, max_value = 1.5, sampling = 'linear'),
                spectral_radius = hp.Float('spectral_radius', min_value = 0.01, max_value = 1.5, sampling = 'linear'),
                leaky = hp.Float('leaky', min_value = 0.01, max_value = 1, sampling = 'linear'))
    model.readout.compile(
        optimizer=keras.optimizers.RMSprop(
            hp.Float('learning_rate',min_value = 1e-5,max_value = 1e-1, sampling = 'log')),
        loss=loss_function,
        metrics=['accuracy'])
    
    return model



def build_model_GRU(hp):
    #GRU
    model = keras.Sequential([
                    keras.layers.Masking(),
                    keras.layers.GRU(hp.Int('units', min_value = 5, max_value = max_units)),
                    keras.layers.Dense(output_units, activation = output_activation)
    ])
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            hp.Float('learning_rate',min_value = 1e-5,max_value = 1e-1, sampling = 'log')),
        loss=loss_function,
        metrics=['accuracy'])
    return model


def build_model_LSTM(hp):
    #LSTM
    model = keras.Sequential([
                    keras.layers.Masking(),
                    keras.layers.LSTM(hp.Int('units', min_value = 5, max_value = max_units)),
                    keras.layers.Dense(output_units, activation = output_activation)
    ])
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            hp.Float('learning_rate',min_value = 1e-5,max_value = 1e-1, sampling = 'log')),
        loss=loss_function,
        metrics=['accuracy'])
    return model


def build_model_ARNN(hp):
    #Anti-symmetric RNN
    model = keras.Sequential([
        keras.layers.Masking(),
        keras.layers.RNN(cell = AntisymmetricRNNCell(
            units = hp.Int('units', min_value = 5, max_value = max_units),
            gamma = hp.Float('gamma', min_value = 0.00001, max_value = 0.1, sampling = 'log'),
            epsilon = hp.Float('epsilon', min_value = 0.00001, max_value = 0.1, sampling = 'log'))),
        keras.layers.Dense(output_units, activation = output_activation)
    ])
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            hp.Float('learning_rate',min_value = 1e-5,max_value = 1e-1, sampling = 'log')),
        loss=loss_function,
        metrics=['accuracy'])
    return model




def build_model_SimpleRNN(hp):
    #Simple (i.e., vanilla) RNN
    model = keras.Sequential([
                    keras.layers.Masking(),
                    keras.layers.SimpleRNN(hp.Int('units', min_value = 5, max_value = max_units)),
                    keras.layers.Dense(output_units, activation = output_activation)
    ])
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            hp.Float('learning_rate',min_value = 1e-5,max_value = 1e-1, sampling = 'log')),
        loss=loss_function,
        metrics=['accuracy'])
    return model

build_model = {
    'EuSN': build_model_EuSN,
    'ESN':build_model_ESN,
    'RESN':build_model_RESN,
    'AntisymmetricRNN':build_model_ARNN,
    'SimpleRNN':build_model_SimpleRNN,
    'LSTM':build_model_LSTM,
    'GRU':build_model_GRU
}


def run_experiment(task_name, model_type, x_test, y_test,x_train, x_val, y_train, y_val):
    
    global results_path
    
    model_selection_times_filename = os.path.join(results_path,'model_selection_times_'+model_type+'.p')
    times_filename = os.path.join(results_path,'times_'+model_type+'.p')
    accuracy_filename = os.path.join(results_path,'accuracy_'+model_type+'.p')
    tuner_filename = os.path.join(results_path,'tuner_'+model_type+'.p')
    
    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')
    
    
    tuner = Hyperband(
        build_model[model_type],
        objective='val_loss',
        max_epochs = num_epochs,
        directory=os.path.join(model_selection_path,task_name,model_type),
        project_name='EuSN-NeuCom2022',
        overwrite = True,
        seed = 42
    )
    
    results_logger = open(results_logger_filename,'w')
    results_logger.write('Experiment with '+model_type+' on dataset '+ task_name + ' starting now\n')
    time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('** local time = '+ time_string_start+'\n')

    initial_model_selection_time = time()
    tuner.search(x_train, y_train,
             epochs=num_epochs,
             validation_data = (x_val,y_val),
             batch_size = batch_size,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)]
            )
    elapsed_model_selection_time = time()-initial_model_selection_time

    time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('model selection concluded at local time = '+ time_string+'\n')
    
    
    #choose the best hyper-parameters
    best_model_hp = tuner.get_best_hyperparameters()[0]
    acc_ts = []
    required_time = []
    tf.random.set_seed(42)
    for i in range(num_guesses):
      initial_time = time()
      model = tuner.hypermodel.build(best_model_hp)
      model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size, validation_data = (x_val, y_val),callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience, restore_best_weights = True)])
      _, acc = model.evaluate(x_test,y_test)
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
    with open(tuner_filename, 'wb') as f:
        pickle.dump(tuner, f)     
    
    print('--'+model_type+' on {}--'.format(task_name))
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
    
    if (model_type in ['EuSN','ESN','RESN']):
        model.readout.summary(print_fn=lambda x: results_logger.write(x + '\n'))
    else:
        model.summary(print_fn=lambda x: results_logger.write(x + '\n'))
    tuner.results_summary(1)
    results_logger.close()


def run_all_experiments(task_name):
     task_settings(task_name)
     x_train_all,y_train_all,x_test, y_test,x_train, x_val, y_train, y_val = load_task_data(task_name)
     
     for model_type in ['EuSN','ESN','RESN','GRU','LSTM','AntisymmetricRNN','SimpleRNN']:
         run_experiment(task_name, model_type, x_test, y_test,x_train, x_val, y_train, y_val)
     
