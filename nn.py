'''
Implementation of a 1 layer neural network
'''

import numpy as np
import pdb


class NN(object):
    def __init__(self):
        self.N = 100
        self.K = 3
        self.D = 2

    def load_data(self):
        X = np.zeros((self.N*self.K, self.D))
        Y = np.zeros(self.N*self.K, dtype='uint8')
        for j in xrange(self.K):
            ix = range(self.N*j, self.N*(j+1))
            r = np.linspace(0.0, 1, self.N)
            t = np.linspace(j*4, (j+1)*4, self.N) + np.random.randn(self.N)*0.2
            X[ix] = np.c_[r*np.sin(t), np.cos(t)]
            Y[ix] = j
        return X, Y

    def solve_nn(self):
        print "Loading data....."
        x, y = self.load_data()
        print "Done!!!!!!!"
        h = 100
        w = 0.01 * np.random.randn(self.D, h)
        b = np.zeros((1, h))
        w2 = 0.01 * np.random.randn(h, self.K)
        b2 = np.zeros((1, self.K))
        step_size = 1e-0
        reg = 1e-3
        num_examples = x.shape[0]
        for i in xrange(10000):
            hidden_layer = np.maximum(0, np.dot(x,w) + b)
            scores = np.dot(hidden_layer, w2) + b2
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            correct_probs = -np.log(probs[range(num_examples), y])
            data_loss = np.sum(correct_probs) / num_examples
            reg_loss = 0.5 * reg * np.sum(w*w) + 0.5 * reg * np.sum(w2 * w2)
            loss = data_loss + reg_loss
            if i % 1000 == 0:
                print "iteration %d: loss %f" %(i, loss)
            dscores = probs
            dscores[range(num_examples), y] -= 1
            dscores /= num_examples
            dw2 = np.dot(hidden_layer.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)
            dhidden = np.dot(dscores, w2.T)
            dhidden[hidden_layer <= 0] = 0
            dw = np.dot(x.T, dhidden)
            db = np.sum(dhidden, axis=0, keepdims = True)
            dw2 += reg * w2
            dw += reg * w
            w = w - step_size * dw
            b = b - step_size * db
            w2 = w2 - step_size * dw2
            b2 = b2 - step_size * db2
        hidden_layer = np.maximum(0, np.dot(x, w) + b)
        scores = np.dot(hidden_layer, w2) + b2
        predicted_class = np.argmax(scores, axis=1)
        print "Training Accuracy: %.2f" %(np.mean(predicted_class == y))

if __name__ == "__main__":
    nn_obj = NN()
    nn_obj.solve_nn()
