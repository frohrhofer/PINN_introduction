import tensorflow as tf


class NeuralNetwork(tf.keras.Sequential):
    
    def __init__(self, n_hidden, n_neurons, activation, verbose=True): 
        
        # call parent constructor & build NN
        super().__init__(name='PhysicsInformedNN')      
        self.build_NN(n_hidden, n_neurons, activation)  
        # Output model summary
        if verbose:
            self.summary() 
            
                
    def build_NN(self, n_hidden, n_neurons, activation):
                
        # build input layer
        self.add(tf.keras.Input(shape=(2, ), name="input"))
        # build hidden layers
        for i in range(n_hidden):
            self.add(tf.keras.layers.Dense(units=n_neurons, 
                                           activation=activation,
                                           name=f"hidden_{i}"))
        # build linear output layer
        self.add(tf.keras.layers.Dense(units=1,
                                       activation=None,
                                       name="output"))