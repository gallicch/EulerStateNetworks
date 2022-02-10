import numpy as np
import tensorflow as tf
from tensorflow import keras


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

    def __init__(self, units, output_units, output_activation,
                 input_scaling = 1., bias_scaling = 1.0, recurrent_scaling = 1,
                 epsilon = 0.01, gamma = 0.001, 
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.Masking(),
                    keras.layers.RNN(cell = EulerReservoirCell(units = units,
                                                                       input_scaling = input_scaling,
                                                                       bias_scaling = bias_scaling,
                                                                       recurrent_scaling = recurrent_scaling,
                                                                       epsilon = epsilon,
                                                                       gamma = gamma)),


        ])
        self.readout = keras.Sequential([
            keras.layers. Dense(output_units, activation = output_activation)
        ])   

       
        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout(reservoir_states)
        return output
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        #the same is done on the validation set
        if 'validation_data' in kwargs:
            x_val,y_val = kwargs['validation_data']
            x_val_states = self.reservoir(x_val)
            kwargs['validation_data'] = (x_val_states, y_val)
        
        return self.readout.fit(x_train_states,y,**kwargs)
        
    def evaluate(self, x, y, **kwargs):
        x_train_states = self.reservoir(x)
        return self.readout.evaluate(x_train_states,y,**kwargs)
    



class ESN(keras.Model):
    #Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    
    def __init__(self, units, output_units, output_activation,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.Masking(),
                    keras.layers.RNN(cell = ReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation = output_activation)
        ])   
        
       
        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout(reservoir_states)
        return output
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        #the same is done on the validation set
        if 'validation_data' in kwargs:
            x_val,y_val = kwargs['validation_data']
            x_val_states = self.reservoir(x_val)
            kwargs['validation_data'] = (x_val_states, y_val)
        
        return self.readout.fit(x_train_states,y,**kwargs)
        
    def evaluate(self, x, y, **kwargs):
        x_train_states = self.reservoir(x)
        return self.readout.evaluate(x_train_states,y,**kwargs)
    


class RESN(keras.Model):
    #Implements a Ring-Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with RingReservoirCell,
    # followed by a trainable dense readout layer for classification

    def __init__(self, units, output_units, output_activation,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.Masking(),
                    keras.layers.RNN(cell = RingReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation = output_activation)
        ])   
        
       
        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout(reservoir_states)
        return output
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        #the same is done on the validation set
        if 'validation_data' in kwargs:
            x_val,y_val = kwargs['validation_data']
            x_val_states = self.reservoir(x_val)
            kwargs['validation_data'] = (x_val_states, y_val)
        
        return self.readout.fit(x_train_states,y,**kwargs)
        
    def evaluate(self, x, y, **kwargs):
        x_train_states = self.reservoir(x)
        return self.readout.evaluate(x_train_states,y,**kwargs)
    
