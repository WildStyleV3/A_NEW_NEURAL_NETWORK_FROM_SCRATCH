import numpy as np 
from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
import math as m
#Softmax Activation

layer_outputs = [4.8, 1.21, 2.385]
exp_values = []
for output in layer_outputs:
    exp_values.append(m.exp(output))
print('exponentiated values:')
print(exp_values)

#Normalize values

norm_base = sum(exp_values)
print('Sum of exponentiated values:', norm_base)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print('Normalized exponentiated values:')
print(norm_values)
print('Sum of normalized values:', sum(norm_values))

#We can perfom the same operation using numpy

layer_outputs = [4.8, 1.21, 2.385]
exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)
#Normalize values
norm_values = exp_values / np.sum(exp_values)
print('Normalized exponentiated values:')
print(norm_values)
print('Sum of normalized values:', np.sum(norm_values))


#Softmax activation
class Activation_Softmax:
    #Forward pass
    def forward(self, inputs):
        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

softmax = Activation_Softmax()
softmax.forward([[1,2,3]])
print(softmax.output)














