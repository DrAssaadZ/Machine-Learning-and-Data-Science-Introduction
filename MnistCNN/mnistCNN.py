# Network architecture:
# 6 layer neural network with 3 convolution layers, input layer 28x28x1, output 10 (10 digits)
# Output labels uses one-hot encoding

# input layer               - X[batch, 28, 28]
# 1 conv. layer             - W1[3,3,1,C1] + b1[C1]
#                             Y1[batch, 28, 28, C1]
# 2 conv. layer             - W2[3, 3, C1, C2] + b2[C2]
# 2.1 max pooling filter 2x2, stride 2 - down sample the input (rescale input by 2) 28x28-> 14x14
#                             Y2[batch, 14,14,C2]
# 3 conv. layer             - W3[3, 3, C2, C3]  + b3[C3]
# 3.1 max pooling filter 2x2, stride 2 - down sample the input (rescale input by 2) 14x14-> 7x7
#                             Y3[batch, 7, 7, C3]
# 4 fully connecteed layer  - W4[7*7*C3, FC4]   + b4[FC4]
#                             Y4[batch, FC4]
# 4t fully connecteed layer  - W4t[FC4, FC5]   + b4t[FC5]
#                             Y4[batch, FC4]
# 5 output layer            - W5[FC5, 10]   + b5[10]
# One-hot encoded labels      Y5[batch, 10]


import visualizations as vis
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

NUM_ITERS = 5000
DISPLAY_STEP = 100
BATCH = 100


# Download images and labels
mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)

# mnist.test (10K images+labels) -> mnist.test.images, mnist.test.labels
# mnist.train (60K images+labels) -> mnist.train.images, mnist.test.labels

# Placeholder for input images, each data sample is 28x28 grayscale images
# All the data will be stored in X - tensor, 4 dimensional matrix
# The first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])


# layers sizes
C1 = 4
C2 = 8
C3 = 16

# fully connected layers
FC4 = 256
FC5 = 128

# weights - initialized with random values from normal distribution mean=0, stddev=0.1

W1 = tf.Variable(tf.truncated_normal([3, 3, 1, C1], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([C1], stddev=0.1))

W2 = tf.Variable(tf.truncated_normal([3, 3, C1, C2], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([C2], stddev=0.1))

W3 = tf.Variable(tf.truncated_normal([3, 3, C2, C3], stddev=0.1))
b3 = tf.Variable(tf.truncated_normal([C3], stddev=0.1))

# first fully connected layer, we have to reshpe previous output to one dim,
W4 = tf.Variable(tf.truncated_normal([7 * 7 * C3, FC4], stddev=0.1))
b4 = tf.Variable(tf.truncated_normal([FC4], stddev=0.1))

# second fully connected layer
W4t = tf.Variable(tf.truncated_normal([FC4, FC5], stddev=0.1))
b4t = tf.Variable(tf.truncated_normal([FC5], stddev=0.1))

# output softmax layer (10 digits)
W5 = tf.Variable(tf.truncated_normal([FC5, 10], stddev=0.1))
b5 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

# flatten the images, unroll each image row by row, create vector[784]

XX = tf.reshape(X, [-1, 784])

# Define the model
stride = 1
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + b1)

k = 2
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + b2)
Y2 = tf.nn.max_pool(Y2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + b3)
Y3 = tf.nn.max_pool(Y3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * C3])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)
Y5 = tf.nn.relu(tf.matmul(Y4, W4t) + b4t)

Ylogits = tf.matmul(Y5, W5) + b5
Y = tf.nn.softmax(Ylogits)


# loss function: cross-entropy = - sum( Y_i * log(Yi) )
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 100.0  # normalized for batches of 100 images,

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training,
learning_rate = 0.003
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Initializing the variables
init = tf.global_variables_initializer()

train_losses = list()
train_acc = list()
test_losses = list()
test_acc = list()

# calculating the labels for the confusion matrix
labels = tf.Variable(tf.zeros([10000.]))
labels = tf.argmax(mnist.test.labels, 1)

# calculating the predictions for the confusion matrix
prediction = tf.argmax(Y, 1)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for i in range(NUM_ITERS + 1):
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(BATCH)

        if i % DISPLAY_STEP == 0:
            # compute training values for visualization
            acc_trn, loss_trn = sess.run([accuracy, cross_entropy],
                                               feed_dict={X: batch_X, Y_: batch_Y})
            # compute testing values for visualization
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy],
                                         feed_dict={X: mnist.test.images, Y_: mnist.test.labels})

            print(
                "#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i, acc_trn, loss_trn, acc_tst, loss_tst))

            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            test_losses.append(loss_tst)
            test_acc.append(acc_tst)

        # the back-propagation training step
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

    # calculating the confusion matrix
    conf_mat_heatmap = sess.run(tf.confusion_matrix(labels=labels, predictions=prediction.eval(feed_dict={X: mnist.test.images})))
    print(conf_mat_heatmap)
    title = "MNIST_Digit recognition in CNN"
    vis.losses_accuracies_plots(conf_mat_heatmap, train_losses, train_acc, test_losses, test_acc, title, DISPLAY_STEP)


