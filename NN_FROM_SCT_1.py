#Coding our first Neuron HANDCODED Neural Network from scratch.
import numpy as np
import matplotlib.pyplot as plt


inputs = [1,2,3,2.5] #This is a vector
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2
bias2 = 3
bias3 = 0.5
output = (inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias)
#print(output)

weights3 = [-0.26, -0.27, 0.17, 0.87]
weights2 = [0.5, -0.91, 0.26, -0.5]

output = ([inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias], 
          [inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2],
          [inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]) 
print(output)

inpputs = [1,2,3,2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2
output = np.dot(weights,inputs) + bias
print(output)



