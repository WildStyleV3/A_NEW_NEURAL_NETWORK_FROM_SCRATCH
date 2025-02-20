import numpy as np 
from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
#How to the ReLu activation looks like

#ReLU activation function in Raw Python

inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = []
for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)
print(output)

inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = []
for i in inputs:
    output.append(max(0,i))
print(output)

inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = np.maximum(0,inputs)
print(output)

############################################################################################################

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        #Initialize weights and biases
        self.weights = 0.01 *np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        
    def forward(self,inputs):
        #Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs,self.weights) + self.biases 
#ReLu Avtivacion as an Object
class Activation_ReLU:
    #Forward Pass
    def forward(self, inputs):
        #Calculate outputs values from inputs
        self.output = np.maximum(0,inputs)

X,y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense1.forward(X)
activation1.forward(dense1.output)

print(activation1.output[:5])






