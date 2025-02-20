import numpy as np 
from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt

#Dense Layer
#Dense layer is a fully connected layer, meaning that each neuron in the layer is connected to every neuron in the previous layer.
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        #Initialize weights and biases
        self.weights = 0.01 *np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        
    def forward(self,inputs):
        #Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs,self.weights) + self.biases    
        
#Create dataset
X, y = spiral_data(samples=100, classes=3)
#Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)   
#Perform a forward pass of our training data through this layer
dense1.forward(X)   
#Let's see output of the first few samples:
print(dense1.output[:5])























