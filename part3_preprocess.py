import tensorflow as tf
import numpy as np
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1


# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes
# grads = tf.placeholder(tf.float32, [None, 784])

# Set model weights
W1 = tf.Variable(tf.random_normal([784, 300], mean=0, stddev=1))
b1 = tf.Variable(tf.random_normal([300], mean=0, stddev = 1))

# #W2 = tf.Variable(tf.random_normal([300, 100], mean=0, stddev= 1))
# #b2 = tf.Variable(tf.random_normal([100], mean=0, stddev= 1))

W3 = tf.Variable(tf.zeros([300, 10]))
b3 = tf.Variable(tf.zeros([10]))

# Construct model

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1); #first hidden layer

#hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2); #second hidden layer

pred = tf.nn.softmax(tf.matmul(hidden1, W3) + b3) # Softmax layer outputs prediction probabilities

# Minimize error using cross entropy 
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# change to training data

yt = mnist.train.labels
xt = mnist.train.images

# grad_W, grad_b = tf.gradients(xs=[W1, b1], ys=cost)
eps = [1, 5, 10, 20, 30, 40, 50]
grad_x = tf.gradients(xs=x, ys=cost)

x_prime1 = tf.clip_by_value(x + eps[0] * tf.sign(grad_x)/256,0,1)
x_prime5 = tf.clip_by_value(x + eps[1] * tf.sign(grad_x)/256,0,1)
x_prime10 = tf.clip_by_value(x + eps[2] * tf.sign(grad_x)/256,0,1)
x_prime20 = tf.clip_by_value(x + eps[3] * tf.sign(grad_x)/256,0,1)
x_prime30 = tf.clip_by_value(x + eps[4] * tf.sign(grad_x)/256,0,1)
x_prime40 = tf.clip_by_value(x + eps[5] * tf.sign(grad_x)/256,0,1)
x_prime50 = tf.clip_by_value(x + eps[6] * tf.sign(grad_x)/256,0,1)
# x_prime = (x + tf.sign(grad_x)/256)
# new_W = W1.assign(W1 - tf.sign(grad_W))
# new_b = b1.assign(b1 - tf.sign(grad_b))

saver = tf.train.Saver()

# Start training
with tf.Session() as sess:
    
    saver.restore(sess, "./checkpoints/trained_model.ckpt")
    print("Model restored.")

    
#     sess.run(tf.global_variables_initializer())
#         # Fit training using batch data
    a1, b1 = sess.run([x_prime1 ,cost], feed_dict={x: xt, y: yt})
    a5, b5 = sess.run([x_prime5 ,cost], feed_dict={x: xt, y: yt})
    a10, b10 = sess.run([x_prime10 ,cost], feed_dict={x: xt, y: yt})
    a20, b20 = sess.run([x_prime20 ,cost], feed_dict={x: xt, y: yt})
    a30, b30 = sess.run([x_prime30 ,cost], feed_dict={x: xt, y: yt})
    a40, b40 = sess.run([x_prime40 ,cost], feed_dict={x: xt, y: yt})
    a50, b50 = sess.run([x_prime50 ,cost], feed_dict={x: xt, y: yt})
    
x_new1 = a1[0,:,:]
x_new5 = a5[0,:,:]
x_new10 = a10[0,:,:]
x_new20 = a20[0,:,:]
x_new30 = a30[0,:,:]
x_new40 = a40[0,:,:]
x_new50 = a50[0,:,:]

np.save('./checkpoints/x_newtrain1', x_new1)
np.save('./checkpoints/x_newtrain5', x_new5)
np.save('./checkpoints/x_newtrain10', x_new10)
np.save('./checkpoints/x_newtrain20', x_new20)
np.save('./checkpoints/x_newtrain30', x_new30)
np.save('./checkpoints/x_newtrain40', x_new40)
np.save('./checkpoints/x_newtrain50', x_new50)