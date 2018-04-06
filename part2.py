import sys
# print( "Name of the script: ", sys.argv[0])

eps_current = int(sys.argv[1])
print ("Input eps =" , int(sys.argv[1]))

import tensorflow as tf
import numpy as np
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def attack_success_rate(xts, xts_new, yts):

        prediction_old = tf.argmax(pred,1)
        prediction_old = prediction_old.eval({x: xts})    
        
        correct_prediction = tf.equal(prediction_old, tf.argmax(yts, 1))
        correct_prediction = correct_prediction.eval({x: xts})
        
        correct_prediction_index = np.where(correct_prediction)
        
        xts_correct = xts_new[correct_prediction_index,:]
        xts_correct = xts_correct[0,:,:]
        
        correct_prediction = correct_prediction[correct_prediction_index]
        prediction_old = prediction_old[correct_prediction_index]
        
        # Result of new test data
        prediction_new = tf.argmax(pred,1)
        prediction_new = prediction_new.eval({x:xts_correct})
        
        # Vind out which index of correct_predictions are changed after perturb
        attack_success_index = np.not_equal(prediction_old, prediction_new)
        
        # Calculate attack ratio
        attack_success_no = np.count_nonzero(attack_success_index)
        correct_prediction_no = np.count_nonzero(correct_prediction)
        
        attack_success_rate = attack_success_no/correct_prediction_no
        
        return attack_success_rate

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

yt = mnist.test.labels
xt = mnist.test.images

def shift(l, n):
    return l[-1:] + l[:-1]

yt_shifted = np.empty_like(yt)

for i in range (np.shape(yt)[0]):
    yt_shifted[i, :] = (shift(yt[i,:].tolist(),1))

# grad_W, grad_b = tf.gradients(xs=[W1, b1], ys=cost)
eps = [1, 5, 10, 20, 30, 40, 50]
grad_x = tf.gradients(xs=x, ys=cost)

x_prime1 = tf.clip_by_value(x - eps[0] * tf.sign(grad_x)/256,0,1)
x_prime5 = tf.clip_by_value(x - eps[1] * tf.sign(grad_x)/256,0,1)
x_prime10 = tf.clip_by_value(x - eps[2] * tf.sign(grad_x)/256,0,1)
x_prime20 = tf.clip_by_value(x - eps[3] * tf.sign(grad_x)/256,0,1)
x_prime30 = tf.clip_by_value(x - eps[4] * tf.sign(grad_x)/256,0,1)
x_prime40 = tf.clip_by_value(x - eps[5] * tf.sign(grad_x)/256,0,1)
x_prime50 = tf.clip_by_value(x - eps[6] * tf.sign(grad_x)/256,0,1)


saver = tf.train.Saver()

# Start training
with tf.Session() as sess:
    
    saver.restore(sess, "./checkpoints/trained_model.ckpt")
    print("Model restored.")

    a1, b1 = sess.run([x_prime1 ,cost], feed_dict={x: xt, y: yt_shifted})
    a5, b5 = sess.run([x_prime5 ,cost], feed_dict={x: xt, y: yt_shifted})
    a10, b10 = sess.run([x_prime10 ,cost], feed_dict={x: xt, y: yt_shifted})
    a20, b20 = sess.run([x_prime20 ,cost], feed_dict={x: xt, y: yt_shifted})
    a30, b30 = sess.run([x_prime30 ,cost], feed_dict={x: xt, y: yt_shifted})
    a40, b40 = sess.run([x_prime40 ,cost], feed_dict={x: xt, y: yt_shifted})
    a50, b50 = sess.run([x_prime50 ,cost], feed_dict={x: xt, y: yt_shifted})
    
x_new1 = a1[0,:,:]
x_new5 = a5[0,:,:]
x_new10 = a10[0,:,:]
x_new20 = a20[0,:,:]
x_new30 = a30[0,:,:]
x_new40 = a40[0,:,:]
x_new50 = a50[0,:,:]

# np.save('./checkpoints/x_new1_p2', x_new1)
# np.save('./checkpoints/x_new5_p2', x_new5)
# np.save('./checkpoints/x_new10_p2', x_new10)
# np.save('./checkpoints/x_new20_p2', x_new20)
# np.save('./checkpoints/x_new30_p2', x_new30)
# np.save('./checkpoints/x_new40_p2', x_new40)
# np.save('./checkpoints/x_new50_p2', x_new50)

# x_new1 = np.load('./checkpoints/x_new1_p2.npy')
# x_new5 = np.load('./checkpoints/x_new5_p2.npy')
# x_new10 = np.load('./checkpoints/x_new10_p2.npy')
# x_new20 = np.load('./checkpoints/x_new20_p2.npy')
# x_new30 = np.load('./checkpoints/x_new30_p2.npy')
# x_new40 = np.load('./checkpoints/x_new40_p2.npy')
# x_new50 = np.load('./checkpoints/x_new50_p2.npy')

with tf.Session() as sess:
    
    saver.restore(sess, "./checkpoints/trained_model.ckpt")
    print("Model restored.")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    if eps_current == 1:
        print ("Eps=1:\t Accuracy:", "{:10.4f}".format(accuracy.eval({x: x_new1[:3000], y: yt[:3000]})), 
               "\t Attack Success Rate:\t", attack_success_rate(xt,x_new1,yt))
    if eps_current == 5:
        print ("Eps=5:\t Accuracy:", "{:10.4f}".format(accuracy.eval({x: x_new5[:3000], y: yt[:3000]})),
               "\t Attack Success Rate:\t", attack_success_rate(xt,x_new5,yt))
    if eps_current == 10:
        print ("Eps=10:\t Accuracy:", "{:10.4f}".format(accuracy.eval({x: x_new10[:3000], y: yt[:3000]})),
               "\t Attack Success Rate:\t", attack_success_rate(xt,x_new10,yt))
    if eps_current == 20:
        print ("Eps=20:\t Accuracy:", "{:10.4f}".format(accuracy.eval({x: x_new20[:3000], y: yt[:3000]})),
               "\t Attack Success Rate:\t", attack_success_rate(xt,x_new20,yt))
    if eps_current == 30:
        print ("Eps=30:\t Accuracy:", "{:10.4f}".format(accuracy.eval({x: x_new30[:3000], y: yt[:3000]})),
               "\t Attack Success Rate:\t", attack_success_rate(xt,x_new10,yt))
    if eps_current == 40:
        print ("Eps=40:\t Accuracy:", "{:10.4f}".format(accuracy.eval({x: x_new40[:3000], y: yt[:3000]})),
               "\t Attack Success Rate:\t", attack_success_rate(xt,x_new10,yt))
    if eps_current == 50:
        print ("Eps=50:\t Accuracy:", "{:10.4f}".format(accuracy.eval({x: x_new50[:3000], y: yt[:3000]})),
               "\t Attack Success Rate:\t", attack_success_rate(xt,x_new10,yt))


import matplotlib.pyplot as plt

print("Saving 100 images for each eps")
for i in range (0,100):
    plt.imsave('./images/part2/eps1_'+str(i)+'.png', x_new1[i].reshape((28,28)))
    plt.imsave('./images/part2/eps5_'+str(i)+'.png', x_new5[i].reshape((28,28)))
    plt.imsave('./images/part2/eps10_'+str(i)+'.png', x_new10[i].reshape((28,28)))
    plt.imsave('./images/part2/eps20_'+str(i)+'.png', x_new20[i].reshape((28,28)))
    plt.imsave('./images/part2/eps30_'+str(i)+'.png', x_new30[i].reshape((28,28)))
    plt.imsave('./images/part2/eps40_'+str(i)+'.png', x_new40[i].reshape((28,28)))
    plt.imsave('./images/part2/eps50_'+str(i)+'.png', x_new50[i].reshape((28,28)))