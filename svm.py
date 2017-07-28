'''
Implementation of Support Vector Machine
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

class SVM(object):
	def __init__(self):
	    pass

	# Loss(i) = sigma(max(0, (wi - wj + delta)))
	def compute_hinge_loss_using_svm(self, x, y, w):
	    delta = 0.0
	    scores = w.dot(x)
	    correct_class_scores, loss = scores[y], 0.0
	    for j in xrange(w.shape[0]):
		if j == y: continue
		loss  = loss + max(0, scores[j] - correct_class_scores + delta)
	    return loss

        def calc_loss(self, xtr, xtr, w):
            bestloss = float("inf")
            bestw = None
            w = np.random.randn(100, 3073) * 0.0001
            for i in xrange(xtr.shape[0]):
                x, y = xtr[i], ytr[i]
                loss = self.compute_hinge_loss_using_svm(x, y, w)
                if loss < bestloss:
                    bestloss = loss
                    bestw = w
            print "in attempt %d the loss: %f, best: %f" %(num, loss, bestloss)
            bestw = np.delete(bestw, bestw.shape[0]-1, 1)
            return bestw

        def calc_grad_loss(self, xtr, ytr, w):
            w = np.random.randn(100, 3073) * 0.0001
            step = 0.001
            while True:
                weights_grad = self.evaluate_grad_descent(compute_hinge_loss_using_svm, xtr, ytr, w)
                if np.abs(w - weights_grad) < 0.001: break
                w = w - step * weights_grad
            return w

        def solve(self):
            xtr, ytr, xte, yte = load_cifar()
            xtr, ytr, xte, yte = np.array(xtr), np.array(ytr), np.array(xte), np.array(yte)
            xtr = xtr.reshape(xtr.shape[0], 3072)
            xte = xte.reshape(xte.shape[0], 3072)
            xtr = np.append(xtr, np.ones((xtr.shape[0], 1), dtype=int), axis=1)
            if loss_fn == "normal":
                bestw = self.calc_loss(xtr, ytr, w)
            else:
                bestw = self.calc_grad_loss(xtr, ytr, w)
            res = []
            for i in xrange(xte.shape[0]):
                scores = bestw.dot(xte[i])
                yte_predict = np.argmax(scores, axis=0)
                res.append(yte_predict)
            res = np.array(res)
            accuracy = np.mean(res == yte)
            print "Accuracy: %d" %(accuracy)

if __name__ == "__main__":
    s = SVM()
    s.solve("normal")
