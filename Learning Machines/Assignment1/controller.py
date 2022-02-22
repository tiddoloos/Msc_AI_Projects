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

        bias2 = weights[weights1_slice:weights1_slice + 2].reshape(1,2)
        weights2 = weights[weights1_slice + 2:].reshape((self.n_hidden[0],2))
        moves = output1.dot(weights2)+ bias2
        moves = moves[0] * 10
        return(moves)
