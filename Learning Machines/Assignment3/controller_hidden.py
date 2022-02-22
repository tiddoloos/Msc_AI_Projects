import numpy as np

def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))

class Player_controller:
    def __init__(self, hidden_n):
        self.n_hidden = [hidden_n]

    def control(self, inputs, weights):
        bias1 = weights[:self.n_hidden[0]].reshape(1,self.n_hidden[0])
        weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
        weights1 = weights[self.n_hidden[0]:weights1_slice].reshape((len(inputs),self.n_hidden[0]))
        output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

        bias2 = weights[weights1_slice:weights1_slice + 3].reshape(1,3)
        weights2 = weights[weights1_slice + 3:].reshape((self.n_hidden[0],3))
        output = output1.dot(weights2)+ bias2
        move = np.argmax(output)
        return(move)
