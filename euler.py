"""
Main file of the repository with the main class definitions

@author: gallicch
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import RidgeClassifier

class EulerReservoirCell(keras.layers.Layer):
    #Implements the reservoir layer of the Euler State Network
    # - the state transition function is achieved by Euler discretization of an ODE
    # - the recurrent weight matrix is constrained to have an anti-symmetric (i.e., skew-symmetric) structure
    
    def __init__(self, units, 
                 input_scaling = 1., bias_scaling = 1.0, recurrent_scaling = 1,
                 epsilon = 0.01, gamma = 0.001, 
                 activation = tf.nn.tanh,
                 **kwargs):
        
        
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.recurrent_scaling = recurrent_scaling
        self.bias_scaling = bias_scaling
        
        self.epsilon = epsilon
        self.gamma = gamma

        self.activation = activation
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        #build the recurrent weight matrix
        I = tf.linalg.eye(self.units)       
        W = tf.random.uniform(shape = (self.units, self.units), minval = -self.recurrent_scaling,maxval = self.recurrent_scaling)
        self.recurrent_kernel = (W - tf.transpose(W) - self.gamma* I)

        #build the input weight matrix
        self.kernel = tf.random.uniform(shape = (input_shape[-1], self.units), minval = -self.input_scaling, maxval = self.input_scaling)
              
        #bias vector
        self.bias = tf.random.uniform(shape = (self.units,), minval = -self.bias_scaling, maxval = self.bias_scaling)
        
        self.built = True


    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation!=None:
            output = prev_output + self.epsilon * self.activation(input_part+ self.bias+ state_part)
        else:
            output = prev_output + self.epsilon * (input_part+ self.bias+ state_part)
        
        return output, [output]
    
    

class ReservoirCell(keras.layers.Layer):
#builds a reservoir as a hidden dynamical layer for a recurrent neural network

    def __init__(self, units, 
                 input_scaling = 1.0, bias_scaling = 1.0,
                 spectral_radius =0.99, 
                 leaky = 1, activation = tf.nn.tanh,
                 **kwargs):
        self.units = units 
        self.state_size = units
        self.input_scaling = input_scaling 
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky #leaking rate
        self.activation = activation
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        #build the recurrent weight matrix
        #uses circular law to determine the values of the recurrent weight matrix
        #rif. paper 
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli. 
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
        value  = (self.spectral_radius / np.sqrt(self.units)) * (6/np.sqrt(12))
        W = tf.random.uniform(shape = (self.units, self.units), minval = -value,maxval = value)
        self.recurrent_kernel = W   
        
        #build the input weight matrix
        self.kernel = tf.random.uniform(shape = (input_shape[-1], self.units), minval = -self.input_scaling, maxval = self.input_scaling)
                         
        #initialize the bias 
        self.bias = tf.random.uniform(shape = (self.units,), minval = -self.bias_scaling, maxval = self.bias_scaling)
        
        self.built = True


    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation!=None:
            output = prev_output * (1-self.leaky) + self.activation(input_part+ self.bias+ state_part) * self.leaky
        else:
            output = prev_output * (1-self.leaky) + (input_part+ self.bias+ state_part) * self.leaky
        
        return output, [output]
    
    
    
class RingReservoirCell(keras.layers.Layer):
#builds a ring reservoir as a hidden dynamical layer for a recurrent neural network
#differently from a conventional reservoir layer, in this case the units in the recurrent
#layer are organized to form a cycle (i.e., a ring)


    def __init__(self, units, 
                 input_scaling = 1.0, bias_scaling = 1.0,
                 spectral_radius =0.99, 
                 leaky = 1, activation = tf.nn.tanh,
                 **kwargs):
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.activation = activation
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
              
        #build the recurrent weight matrix
        I = tf.linalg.eye(self.units)
        W = self.spectral_radius * tf.concat([I[:,-1:],I[:,0:-1]],axis = 1)
        self.recurrent_kernel = W               
        
        #build the input weight matrix
        self.kernel = tf.random.uniform(shape = (input_shape[-1], self.units), minval = -self.input_scaling, maxval = self.input_scaling)
        
        
        self.bias = tf.random.uniform(shape = (self.units,), minval = -self.bias_scaling, maxval = self.bias_scaling)
        
        self.built = True


    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation!=None:
            output = prev_output * (1-self.leaky) + self.activation(input_part+ self.bias+ state_part) * self.leaky
        else:
            output = prev_output * (1-self.leaky) + (input_part+ self.bias+ state_part) * self.leaky
        
        return output, [output]


class AntisymmetricRNNCell(keras.layers.Layer):
    #Implements the recurrent layer of an Antisymmetric RNN
    #from the paper Chang, Bo, et al. "AntisymmetricRNN: A dynamical system view 
    #on recurrent neural networks." arXiv preprint arXiv:1902.09689 (2019).

    def __init__(self, units, 
                 input_scaling = 1.,
                 epsilon = 0.01, gamma = 0.001, 
                 activation = tf.nn.tanh,
                 **kwargs):
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        
        self.epsilon = epsilon
        self.gamma = gamma

        self.activation = activation
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        #build the weight matrices
        self.recurrent_kernel = self.add_weight(name = 'recurrent kernel', shape = (self.units, self.units), initializer = 'orthogonal')
        self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-1], self.units), initializer = 'glorot_normal')
        self.bias = self.add_weight(name = 'bias', shape = (self.units,), initializer = 'zeros')
     
        self.built = True


    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        
        state_part = tf.matmul(prev_output, (self.recurrent_kernel- tf.transpose(self.recurrent_kernel)-self.gamma * tf.linalg.eye(self.units)))
        if self.activation!=None:
            output = prev_output + self.epsilon * self.activation(input_part+ self.bias+ state_part)
        else:
            output = prev_output + self.epsilon * (input_part+ self.bias+ state_part)
        
        return output, [output]
    
    
    

    
    
    
class EuSN(keras.Model):
    #Implements an Euler State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with EulerReservoirCell,
    # followed by a trainable dense readout layer for classification

    def __init__(self, units,
                 input_scaling = 1., bias_scaling = 1.0, recurrent_scaling = 1,
                 epsilon = 0.01, gamma = 0.001, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = EulerReservoirCell(units = units,
                                                                       input_scaling = input_scaling,
                                                                       bias_scaling = bias_scaling,
                                                                       recurrent_scaling = recurrent_scaling,
                                                                       epsilon = epsilon,
                                                                       gamma = gamma)),


        ])
        
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

       
        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output

    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
       
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)



class ESN(keras.Model):
    #Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    
    def __init__(self, units,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = ReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
       
        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)
    


class RESN(keras.Model):
    #Implements a Ring-Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with RingReservoirCell,
    # followed by a trainable dense readout layer for classification

    
    def __init__(self, units,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = RingReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
       
        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)

class DeepESN(keras.Model):
    # Implements a Deep Echo State Network model for time-series classification problems
    #
    # The architecture comprises several recurrent layers with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    # reference paper: Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli. "Deep reservoir computing: A critical experimental analysis." Neurocomputing 268 (2017): 87-99.
    
    def __init__(self, units,
                 num_layers = 3,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        #finds how many units are needed in each layer
        units_per_layer = int(np.floor(units / num_layers))
        units_first_layer = int(units - (units_per_layer * (num_layers-1)))
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = ReservoirCell(units = units_first_layer,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky),
                                    return_sequences = True, return_state = True)
        ])
        
        #adds the pipeline of reservoir layers into the architecture
        #notice that all the layers share the same value of the spectral radius
        #and the same values for the input and bias scaling across the layers
        for i in range(num_layers-2):
            self.reservoir.add(
                keras.layers.RNN(cell = ReservoirCell(units = units_per_layer,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky),
                                return_sequences = True, return_state = True)
            )
        self.reservoir.add(
            keras.layers.RNN(cell = ReservoirCell(units = units_per_layer,
                                                      input_scaling = input_scaling,
                                                      bias_scaling = bias_scaling,
                                                      spectral_radius = spectral_radius,
                                                      leaky = leaky),
                            return_sequences = False)
        )
        
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
       
        
    def call(self, inputs):       
        x = inputs
        for i in range(len(model.reservoir.layers)-1):
            allstates,r = model.reservoir.layers[i](x)
            if (i==0):
                reservoir_states = r
            else:
                reservoir_states = np.concatenate((reservoir_states,r), axis = 1)
            x = allstates
        r = model.reservoir.layers[-1](x)
        reservoir_states = np.concatenate((reservoir_states,r), axis = 1)
        
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        #x_train_states = self.reservoir(x)
        
        xr = x
        for i in range(len(self.reservoir.layers)-1):
            allstates,r = self.reservoir.layers[i](xr)
            if (i==0):
                x_train_states = r
            else:
                x_train_states = np.concatenate((x_train_states,r), axis = 1)
            xr = allstates
        r = self.reservoir.layers[-1](xr)
        x_train_states = np.concatenate((x_train_states,r), axis = 1)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        xr = x
        for i in range(len(self.reservoir.layers)-1):
            allstates,r = self.reservoir.layers[i](xr)
            if (i==0):
                x_train_states = r
            else:
                x_train_states = np.concatenate((x_train_states,r), axis = 1)
            xr = allstates
        r = self.reservoir.layers[-1](xr)
        x_train_states = np.concatenate((x_train_states,r), axis = 1)
        return self.readout.score(x_train_states,y)
    
#### The following variants use a buffer to avoid computing all the states at once 
#### use the following for larger datasets (as in the case of sequential MNIST used in the paper)
    
class EuSN_buffer(keras.Model):
    #Implements an Euler State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with EulerReservoirCell,
    # followed by a trainable dense readout layer for classification

    def __init__(self, units,
                 input_scaling = 1., bias_scaling = 1.0, recurrent_scaling = 1,
                 epsilon = 0.01, gamma = 0.001, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 buffer_size = 128,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.units = units
        self.buffer_size = buffer_size
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = EulerReservoirCell(units = units,
                                                                       input_scaling = input_scaling,
                                                                       bias_scaling = bias_scaling,
                                                                       recurrent_scaling = recurrent_scaling,
                                                                       epsilon = epsilon,
                                                                       gamma = gamma)),


        ])
        
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

       
        
    def call(self, x):
        #training set
        buffer_size = self.buffer_size
        buffer_number = np.int(np.ceil(x.shape[0] / buffer_size))

        x_train_1 = np.zeros(shape = (x.shape[0],self.units))
        for i in range(buffer_number-1):
            xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        i = i+1
        xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
        x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        
        
        output = self.readout.predict(x_train_1)

        return output
    
    
        
    def evaluate(self, x, y):
        #training set
        buffer_size = self.buffer_size
        buffer_number = np.int(np.ceil(x.shape[0] / buffer_size))

        x_train_1 = np.zeros(shape = (x.shape[0],self.units))
        for i in range(buffer_number-1):
            xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        i = i+1
        xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
        x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        
        
        return self.readout.score(x_train_1,y)
    
    def fit(self, x, y, **kwargs):
        #training set
        buffer_size = self.buffer_size
        buffer_number = np.int(np.ceil(x.shape[0] / buffer_size))

        x_train_1 = np.zeros(shape = (x.shape[0],self.units))
        for i in range(buffer_number-1):
            xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        i = i+1
        xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
        x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
       
        
        return self.readout.fit(x_train_1,y)
    
    
class ESN_buffer(keras.Model):
    #Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    
    def __init__(self, units,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 buffer_size = 128,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.units = units
        self.buffer_size = buffer_size
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = ReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
       
        
    def call(self, x):
        #training set
        buffer_size = self.buffer_size
        buffer_number = np.int(np.ceil(x.shape[0] / buffer_size))

        x_train_1 = np.zeros(shape = (x.shape[0],self.units))
        for i in range(buffer_number-1):
            xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        i = i+1
        xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
        x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        
        
        output = self.readout.predict(x_train_1)
        return output
    
    
        
    def evaluate(self, x, y):
        #training set
        buffer_size = self.buffer_size
        buffer_number = np.int(np.ceil(x.shape[0] / buffer_size))

        x_train_1 = np.zeros(shape = (x.shape[0],self.units))
        for i in range(buffer_number-1):
            xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        i = i+1
        xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
        x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        
        return self.readout.score(x_train_1,y)
    
    def fit(self, x, y, **kwargs):
        #training set
        buffer_size = self.buffer_size
        buffer_number = np.int(np.ceil(x.shape[0] / buffer_size))

        x_train_1 = np.zeros(shape = (x.shape[0],self.units))
        for i in range(buffer_number-1):
            xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        i = i+1
        xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
        x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        
        return self.readout.fit(x_train_1,y)
    
    
class RESN_buffer(keras.Model):
    #Implements an Ring-Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with RingReservoirCell,
    # followed by a trainable dense readout layer for classification
    
    def __init__(self, units,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 buffer_size = 128,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.units = units
        self.buffer_size = buffer_size
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = RingReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
       
        
    def call(self, x):
        #training set
        buffer_size = self.buffer_size
        buffer_number = np.int(np.ceil(x.shape[0] / buffer_size))

        x_train_1 = np.zeros(shape = (x.shape[0],self.units))
        for i in range(buffer_number-1):
            xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        i = i+1
        xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
        #rl = reservoir(xlocal)
        x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        
        
        output = self.readout.predict(x_train_1)
        return output
    
    
        
    def evaluate(self, x, y):
        #training set
        buffer_size = self.buffer_size
        buffer_number = np.int(np.ceil(x.shape[0] / buffer_size))

        x_train_1 = np.zeros(shape = (x.shape[0],self.units))
        for i in range(buffer_number-1):
            xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        i = i+1
        xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
        x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        
        return self.readout.score(x_train_1,y)
    
    def fit(self, x, y, **kwargs):
        #training set
        buffer_size = self.buffer_size
        buffer_number = np.int(np.ceil(x.shape[0] / buffer_size))

        x_train_1 = np.zeros(shape = (x.shape[0],self.units))
        for i in range(buffer_number-1):
            xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        i = i+1
        xlocal = x[i*buffer_size:(i+1)*buffer_size,:,:]
        x_train_1[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir(xlocal)
        
        return self.readout.fit(x_train_1,y)
        
class DeepESN_buffer(keras.Model):
    #Implements a Deep Echo State Network model for time-series classification problems
    
    def __init__(self, units,
                 num_layers = 3,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 buffer_size = 128,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.buffer_size = buffer_size
        
        #finds how many units are needed in each layer
        self.units_per_layer = int(np.floor(units / num_layers))
        self.units_first_layer = int(units - (self.units_per_layer * (num_layers-1)))
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = ReservoirCell(units = self.units_first_layer,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky),
                                    return_sequences = True, return_state = True)
        ])
        
        
        for i in range(num_layers-2):
            self.reservoir.add(
                keras.layers.RNN(cell = ReservoirCell(units = self.units_per_layer,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky),
                                return_sequences = True, return_state = True)
            )
        self.reservoir.add(
            keras.layers.RNN(cell = ReservoirCell(units = self.units_per_layer,
                                                      input_scaling = input_scaling,
                                                      bias_scaling = bias_scaling,
                                                      spectral_radius = spectral_radius,
                                                      leaky = leaky),
                            return_sequences = False)
        )
        
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

    def reservoir_buffer(self, x, reservoir_number):
        buffer_size = self.buffer_size
        buffer_number = np.int(np.ceil(x.shape[0] / buffer_size))
        if (reservoir_number == 0):
            num_units = self.units_first_layer
        else:
            num_units = self.units_per_layer
        
        if (reservoir_number < len(self.reservoir.layers)-1):
            x_states_all = np.zeros(shape = (x.shape[0],x.shape[1],num_units))
            x_states_last = np.zeros(shape = (x.shape[0],num_units))
            for i in range(buffer_number-1):
                x_buffer = x[i*buffer_size:(i+1)*buffer_size,:,:]
                x_states_all[i*buffer_size:(i+1)*buffer_size,:,:], x_states_last[i*buffer_size:(i+1)*buffer_size,:]= self.reservoir.layers[reservoir_number](x_buffer)
            i = i+1
            x_buffer = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_states_all[i*buffer_size:(i+1)*buffer_size,:,:], x_states_last[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir.layers[reservoir_number](x_buffer)
            return x_states_all, x_states_last
        else:
            x_states_last = np.zeros(shape = (x.shape[0],num_units))
            for i in range(buffer_number-1):
                x_buffer = x[i*buffer_size:(i+1)*buffer_size,:,:]
                x_states_last[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir.layers[reservoir_number](x_buffer)
            i = i+1
            x_buffer = x[i*buffer_size:(i+1)*buffer_size,:,:]
            x_states_last[i*buffer_size:(i+1)*buffer_size,:] = self.reservoir.layers[reservoir_number](x_buffer)
            return x_states_last

    
        
    def call(self, inputs):
        x = inputs
        for i in range(len(model.reservoir.layers)-1):
            allstates,r = self.reservoir_buffer(x,i)
            if (i==0):
                reservoir_states = r
            else:
                reservoir_states = np.concatenate((reservoir_states,r), axis = 1)
            x = allstates
        r = self.reservoir_buffer(x,i+1)
        reservoir_states = np.concatenate((reservoir_states,r), axis = 1)
        
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        
        xr = x
        for i in range(len(self.reservoir.layers)-1):
            print('x')
            allstates,r = self.reservoir_buffer(xr,i)
            if (i==0):
                x_train_states = r
            else:
                x_train_states = np.concatenate((x_train_states,r), axis = 1)
            xr = allstates
        r = self.reservoir_buffer(xr,i+1)
        x_train_states = np.concatenate((x_train_states,r), axis = 1)
               
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        #x_train_states = self.reservoir(x)
        xr = x
        for i in range(len(self.reservoir.layers)-1):
            allstates,r = self.reservoir_buffer(xr,i)
            if (i==0):
                x_train_states = r
            else:
                x_train_states = np.concatenate((x_train_states,r), axis = 1)
            xr = allstates
        r = self.reservoir_buffer(xr,i+1)
        x_train_states = np.concatenate((x_train_states,r), axis = 1)
        return self.readout.score(x_train_states,y)