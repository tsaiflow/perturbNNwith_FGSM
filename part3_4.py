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

x_new1 = np.load('./checkpoints/x_newtrain1.npy')
x_new5 = np.load('./checkpoints/x_newtrain5.npy')
x_new10 = np.load('./checkpoints/x_newtrain10.npy')
x_new20 = np.load('./checkpoints/x_newtrain20.npy')
x_new30 = np.load('./checkpoints/x_newtrain30.npy')
x_new40 = np.load('./checkpoints/x_newtrain40.npy')
x_new50 = np.load('./checkpoints/x_newtrain50.npy')

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W1 = tf.Variable(tf.random_normal([784, 300], mean=0, stddev=1))
b1 = tf.Variable(tf.random_normal([300], mean=0, stddev = 1))

#W2 = tf.Variable(tf.random_normal([300, 100], mean=0, stddev= 1))
#b2 = tf.Variable(tf.random_normal([100], mean=0, stddev= 1))

W3 = tf.Variable(tf.zeros([300, 10]))
b3 = tf.Variable(tf.zeros([10]))


# Construct model

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1); #first hidden layer

#hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2); #second hidden layer

pred = tf.nn.softmax(tf.matmul(hidden1, W3) + b3) # Softmax layer outputs prediction probabilities

# Minimize error using cross entropy 

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# optimizer = {}
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# for name, fcn in cost:
#     optimizer["optimizer_{0}".format(name)] = tf.train.GradientDescentOptimizer(learning_rate).minimize(fcn)


x_comb1 = np.append(mnist.train.images, x_new1, axis = 0)
x_comb5 = np.append(mnist.train.images, x_new5, axis = 0)
x_comb10 = np.append(mnist.train.images, x_new10, axis = 0)
x_comb20 = np.append(mnist.train.images, x_new20, axis = 0)
x_comb30 = np.append(mnist.train.images, x_new30, axis = 0)
x_comb40 = np.append(mnist.train.images, x_new40, axis = 0)
x_comb50 = np.append(mnist.train.images, x_new50, axis = 0)

# y_comb = np.append(mnist.train.labels, mnist.test.labels[:3000], axis = 0)

y_comb = np.append(mnist.train.labels, mnist.train.labels, axis = 0)

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[ idx]
    labels_shuffle = labels[ idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def attack_success_rate(xts, xts_new, yts):
        # Result of old test data
        prediction_old = tf.argmax(pred,1)
        prediction_old = prediction_old.eval({x: xts})    
        
        correct_prediction = tf.equal(prediction_old, tf.argmax(yts, 1))
        correct_prediction = correct_prediction.eval({x: xts})
        
        # Because we are only 
        correct_prediction_index = np.where(correct_prediction)
        
        xts_correct = xts_new[correct_prediction_index,:]
        xts_correct = xts_correct[0,:,:]
        
        correct_prediction = correct_prediction[correct_prediction_index]
        prediction_old = prediction_old[correct_prediction_index]
        
        # Result of new test data
        prediction_new = tf.argmax(pred,1)
        prediction_new = prediction_new.eval({x:xts_correct})
        
        #changed_prediction = tf.not_equal(prediction_old, prediction_new)
        #changed_prediction = changed_prediction.eval({x: xts_correct})
        
        # Vind out which index of correct_predictions are changed after perturb
        attack_success_index = np.not_equal(prediction_old, prediction_new)
        
        # Calculaye attack ratio
        attack_success_no = np.count_nonzero(attack_success_index)
        correct_prediction_no = np.count_nonzero(correct_prediction)
        
        attack_success_rate = attack_success_no/correct_prediction_no
        
        return attack_success_rate


saver = tf.train.Saver()
# save_dir = 'checkpoints/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# save_path = os.path.join(save_dir, 'trained_model')

saver = tf.train.Saver()
with tf.Session() as sess:
    
    print("Original model(no retraining).")
    saver.restore(sess, "./checkpoints/trained_model.ckpt")
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    
    # Calculate accuracy for 3000 examples; you should get roughly ~90% accuracy although it might vary from run to run
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     print ("Accuracy of original train dataset:", accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))
    # print ("Accuracy of original test dataset(3000):", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
    # print ("Accuracy of perturbed(eps = 1) train dataset:", accuracy.eval({x: x_new1, y: mnist.train.labels}))
    # print ("Accuracy of perturbed(eps = 5) train dataset:", accuracy.eval({x: x_new5, y: mnist.train.labels}))
    # print ("Accuracy of perturbed(eps = 10) train dataset:", accuracy.eval({x: x_new10, y: mnist.train.labels}))
    # print ("Accuracy of perturbed(eps = 20) train dataset:", accuracy.eval({x: x_new20, y: mnist.train.labels}))
    # print ("Accuracy of perturbed(eps = 30) train dataset:", accuracy.eval({x: x_new30, y: mnist.train.labels}))
    # print ("Accuracy of perturbed(eps = 40) train dataset:", accuracy.eval({x: x_new40, y: mnist.train.labels}))
    # print ("Accuracy of perturbed(eps = 50) train dataset:", accuracy.eval({x: x_new50, y: mnist.train.labels}))
    # print ("Accuracy of mixed normal & perturbed FGSM dataset:", accuracy.eval({x: np.append(x_new10, mnist.train.images, axis = 0), 
    #                                    y: np.append(mnist.train.labels, mnist.train.labels, axis = 0)}))


with tf.Session() as sess:
    if eps_current == 1: 
        print("Retrained eps = 1 model")
        saver.restore(sess, "./checkpoints/trained_model_p1.ckpt")
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        # Calculate accuracy for 3000 examples; you should get roughly ~90% accuracy although it might vary from run to run
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print ("Accuracy of original train dataset:", accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))
        print ("Accuracy of original test dataset(3000):", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        print ("Accuracy of perturbed(eps = 1) train dataset:", accuracy.eval({x: x_new1, y: mnist.train.labels}))
        print ("Accuracy of mixed normal & perturbed FGSM dataset:", accuracy.eval({x: np.append(x_new1, mnist.train.images, axis = 0), 
                                           y: np.append(mnist.train.labels, mnist.train.labels, axis = 0)}))

    if eps_current == 5: 
        print("Retrained eps = 5 model")
        saver.restore(sess, "./checkpoints/trained_model_p5.ckpt")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        # Calculate accuracy for 3000 examples; you should get roughly ~90% accuracy although it might vary from run to run
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #     print ("Accuracy of original train dataset:", accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))
        print ("Accuracy of original test dataset(3000):", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        print ("Accuracy of perturbed(eps = 5) train dataset:", accuracy.eval({x: x_new5, y: mnist.train.labels}))
        print ("Accuracy of mixed normal & perturbed FGSM dataset:", accuracy.eval({x: np.append(x_new5, mnist.train.images, axis = 0), 
                                           y: np.append(mnist.train.labels, mnist.train.labels, axis = 0)}))

    if eps_current == 10: 
        print("Retrained eps = 10 model")
        saver.restore(sess, "./checkpoints/trained_model_p10.ckpt")
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        # Calculate accuracy for 3000 examples; you should get roughly ~90% accuracy although it might vary from run to run
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     print ("Accuracy of original train dataset:", accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))
        print ("Accuracy of original test dataset(3000):", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        print ("Accuracy of perturbed(eps = 10) train dataset:", accuracy.eval({x: x_new10, y: mnist.train.labels}))
        print ("Accuracy of mixed normal & perturbed FGSM dataset:", accuracy.eval({x: np.append(x_new10, mnist.train.images, axis = 0), 
                                           y: np.append(mnist.train.labels, mnist.train.labels, axis = 0)}))

    if eps_current == 20: 
        print("Retrained eps = 20 model")
        saver.restore(sess, "./checkpoints/trained_model_p20.ckpt")
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        # Calculate accuracy for 3000 examples; you should get roughly ~90% accuracy although it might vary from run to run
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     print ("Accuracy of original train dataset:", accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))
        print ("Accuracy of original test dataset(3000):", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        print ("Accuracy of perturbed(eps = 20) train dataset:", accuracy.eval({x: x_new20, y: mnist.train.labels}))
        print ("Accuracy of mixed normal & perturbed FGSM dataset:", accuracy.eval({x: np.append(x_new20, mnist.train.images, axis = 0), 
                                           y: np.append(mnist.train.labels, mnist.train.labels, axis = 0)}))

    if eps_current == 30:        
        print("Retrained eps = 30 model")
        saver.restore(sess, "./checkpoints/trained_model_p30.ckpt")
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        # Calculate accuracy for 3000 examples; you should get roughly ~90% accuracy although it might vary from run to run
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     print ("Accuracy of original train dataset:", accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))
        print ("Accuracy of original test dataset(3000):", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        print ("Accuracy of perturbed(eps = 30) train dataset:", accuracy.eval({x: x_new30, y: mnist.train.labels}))
        print ("Accuracy of mixed normal & perturbed FGSM dataset:", accuracy.eval({x: np.append(x_new30, mnist.train.images, axis = 0), 
                                           y: np.append(mnist.train.labels, mnist.train.labels, axis = 0)}))
    
    if eps_current == 40:     
        print("Retrained eps = 40 model")
        saver.restore(sess, "./checkpoints/trained_model_p40.ckpt")
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        # Calculate accuracy for 3000 examples; you should get roughly ~90% accuracy although it might vary from run to run
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     print ("Accuracy of original train dataset:", accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))
        print ("Accuracy of original test dataset(3000):", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        print ("Accuracy of perturbed(eps = 40) train dataset:", accuracy.eval({x: x_new40, y: mnist.train.labels}))
        print ("Accuracy of mixed normal & perturbed FGSM dataset:", accuracy.eval({x: np.append(x_new40, mnist.train.images, axis = 0), 
                                           y: np.append(mnist.train.labels, mnist.train.labels, axis = 0)}))
    
    if eps_current == 50:     
        print("Retrained eps = 50 model")
        saver.restore(sess, "./checkpoints/trained_model_p50.ckpt")
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        # Calculate accuracy for 3000 examples; you should get roughly ~90% accuracy although it might vary from run to run
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     print ("Accuracy of original train dataset:", accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))
        print ("Accuracy of original test dataset(3000):", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        print ("Accuracy of perturbed(eps = 50) train dataset:", accuracy.eval({x: x_new50, y: mnist.train.labels}))
        print ("Accuracy of mixed normal & perturbed FGSM dataset:", accuracy.eval({x: np.append(x_new50, mnist.train.images, axis = 0), 
                                           y: np.append(mnist.train.labels, mnist.train.labels, axis = 0)}))


import matplotlib.pyplot as plt

print("Saving 100 images for each eps")
for i in range (0,100):
    plt.imsave('./images/part3/eps1'+str(i)+'.png', x_new1[i].reshape((28,28)))
    plt.imsave('./images/part3/eps5'+str(i)+'.png', x_new5[i].reshape((28,28)))
    plt.imsave('./images/part3/eps10'+str(i)+'.png', x_new10[i].reshape((28,28)))
    plt.imsave('./images/part3/eps20'+str(i)+'.png', x_new20[i].reshape((28,28)))
    plt.imsave('./images/part3/eps30'+str(i)+'.png', x_new30[i].reshape((28,28)))
    plt.imsave('./images/part3/eps40'+str(i)+'.png', x_new40[i].reshape((28,28)))
    plt.imsave('./images/part3/eps50'+str(i)+'.png', x_new50[i].reshape((28,28)))