'''
Implementation of KNN
L1 Norm = Sigma(i1 - i2)
L2 Norm = root(Sigm(i1 - i2) ^ 2)
'''

import cPickle
import numpy as np
import pdb

def unpickle(in_file):
    with open(in_file, "rb") as fo:
        dct = cPickle.load(fo)
    return dct

def load_cifar():
    train = unpickle("data/train")
    test  = unpickle("data/test")
    train_data, train_class = train['data'], train['fine_labels']
    test_data, test_class = test['data'], test['fine_labels']
    return train_data, train_class, test_data, test_class

class NearestNeighbour(object):
    def __init__(self):
        pass

    def train(self, x, y):
        self.xtr = x
        self.ytr = y

    def predict(self, x):
        num_test = x.shape[0]
        ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        for i in xrange(num_test):
            if i % 10 == 0: print "processed %d images!!" %(i)
            distances = np.sum(np.abs(self.xtr - x[i,:]), axis=1) # Here we can use L1 Norm (or) L2 Norm
            min_index = np.argmin(distances)
            ypred[i] = self.ytr[min_index]
        return ypred

if __name__ == "__main__":
    nn = NearestNeighbour()
    xtr, ytr, xte, yte = load_cifar()
    xtr, ytr, xte, yte = np.array(xtr), np.array(ytr), np.array(xte), np.array(yte)
    nn.train(xtr, ytr)
    yte_predict = nn.predict(xte)
    print "Accuracy: %f" %(np.mean(yte_predict == yte))
