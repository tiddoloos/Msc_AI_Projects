import numpy as np

def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Player_controller:
    def __init__(self, output):
        self.output = [output]
    
    def control(self, inputs, weights):
        bias = weights[:self.output[0]].reshape(1,self.output[0])
        weights1_slice = len(inputs)*self.output[0] + self.output[0]
        weights1 = weights[self.output[0]:weights1_slice].reshape((len(inputs),self.output[0]))
        output = inputs.dot(weights1) + bias
        move = np.argmax(output)
        return move
