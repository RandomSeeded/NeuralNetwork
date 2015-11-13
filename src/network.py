import numpy as np

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Translation: generate a Yx1 matrix for each Y, where Y is a layer not including input
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Translation: generate a YxX matrix for each Y and X where Y & X are adjacent layers 
        self.weights = [np.random.randn(y, x)
                for x,y in zip(sizes[:-1], sizes[1:])]

    def __repr__(self):
        return 'Biases: '+str(self.biases) + '\n' + 'Weights: '+ str(self.weights)

    def feedforward(self, a):
        """Return the output of the network if "a" is input"""
        # For 2,3,1 network, 
        # biases is a 1x3 and 1x1 matrix (one for each node other than input)
        # and weights is a 2x3 matrix and a 3x1 matrix (one for each connection between layers)
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

# Sigmoid function: creates a curve from 0-1 so that we can have improvements due to weight changes instead of stepwise/binary 0 / 1
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

net = Network([2,3,1])

inArray = np.ndarray(shape=(2,1))
print(net.feedforward(inArray))


