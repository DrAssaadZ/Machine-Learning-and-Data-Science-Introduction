# Vanilla deep network
# get the data: http://ufldl.stanford.edu/housenumbers/
from __future__ import print_function, division
from builtins import range

# Note: you may need to update your version of future
# sudo pip install -U future

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# to load matlab files
from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime

# this function transform a vector into a matrix of N by 10, 10 is number of values when the data is of class y, the column y is 1, the rest are 0
def y2indicator(y):
	N = len(y)
	ind = np.zeros((N,10))
	for i in xrange(N):
		ind[i, y[i]] = 1
	return ind

def error_rate(p, t):
    return np.mean(p != t)

def flatten(X):
    # input will be (32, 32, 3, N)
    # output will be (N, 3072)
    # N is the number of examples(in matlab files), we use N to get it
    N = X.shape[-1]
    flat = np.zeros((N, 3072))
    for i in range(N):
        flat[i] = X[:,:,:,i].reshape(3072)
    return flat

def main():
    train = loadmat('dataset3/train_32x32.mat')
    test = loadmat('dataset3/test_32x32.mat')

    # after loading the data it is devided automaticaly into X and y features
    Xtrain = flatten(train['X'].astype(np.float32) / 255.0)
    Ytrain = train['y'].flatten() - 1			# matlab starts counting from 1 so we sub 1
    Xtrain, Ytrain = shuffle(Xtrain,Ytrain)		# we shuffle it to get different result each time!
    ytrain_ind = y2indicator(Ytrain)

    Xtest = flatten(test['X'].astype(np.float32) / 255.0)
    Ytest = train['y'].flatten() - 1
    ytest_ind = y2indicator(Ytest)

    # gradient descent params
    max_iterations = 20
    print_period = 10
    N, D = Xtrain.shape
    batch_size = 500
    n_batchs = N / batch_size

    # layers params
    M1 = 1000	# 1st layer
    M2 = 500	# 2nd layer
    K = 10		# output layer

    # initializing the weights and biases
    # np.sqrt(D + M1) is a normalisation
    W1_init = np.random.randn(D, M1) / np.sqrt(D + M1)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2 + K)
    b3_init = np.zeros(K)

    # define variables and expressions
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.int32, shape=(None,), name='T')
    
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )
    Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )
    logits = tf.matmul(Z2, W3) + b3

    cost = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=T
        )
    )

    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    # we'll use this to calculate the error rate
    predict_op = tf.argmax(logits, 1)

    t0 = datetime.now()
    LL = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz),]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    LL.append(test_cost)
    print("Elapsed time:", (datetime.now() - t0))
    plt.plot(LL)
    plt.show()

if __name__ == '__main__':
    main()