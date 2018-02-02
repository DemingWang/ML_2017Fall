#!python
# -*- coding: utf-8 -*-#
"""
Softmax Regression for MNIST data.

@author: Bhishan Poudel

@date: Oct 15, 2017

@email: bhishanpdl@gmail.com

Ref: https://ludlows.github.io/2016-08-11-Recognition-MNIST-Handwriting-Digits/
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from six.moves import reduce

def softmax_tf():
        
    train_image = np.load('data/train-image-py.npy')
    train_label = np.load('data/trainlabel-py.npy')
    test_image = np.load('data/t10k-iimage-py.npy')
    test_label = np.load('data/t10k-label-py.npy')
    
    # Reformat data
    image_size = 28
    num_labels = 10
    num_channels = 1 # gray scale

    reformat = lambda data,labels: (data.reshape((-1, image_size, image_size, 1)).astype(np.float32),(np.arange(num_labels) == labels[:,None]).astype(np.float32))
    
    train_dataset, train_labels = reformat(train_image, train_label)
    test_dataset, test_labels = reformat(test_image, test_label)
    
    print('train_dataset size: ', train_dataset.shape) # (60000, 28, 28, 1)
    print('train_labels size: ', train_labels.shape)   # (60000, 10)
    print('test_dataset size: ', test_dataset.shape)   # (10000, 28, 28, 1)
    print('test_labels size: ', test_labels.shape)     # (10000, 10)
    
    
    accuracy = lambda pred, labels: (100.0 * np.sum(np.argmax(pred,1) == np.argmax(labels,1))/pred.shape[0] )
    batch_size = 128

    num_steps = 4501

    graph = tf.Graph()  
    with graph.as_default():  
        # Input data.  
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels)) # num_channels=1 grayscale   
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))  
        tf_test_dataset = tf.constant(test_dataset)  
            
            # Variables.
       
        filter1 = tf.Variable(tf.truncated_normal([1,1,1,6], stddev=0.1))  
        biases1 = tf.Variable(tf.zeros([6]))  
              
        filter2 = tf.Variable(tf.truncated_normal( [5,5,6,16], stddev=0.1))  
        biases2 = tf.Variable(tf.constant(1.0, shape=[16]))  
             
        filter3 = tf.Variable(tf.truncated_normal([5,5, 16, 120], stddev=0.1))  
        biases3 = tf.Variable(tf.constant(1.0, shape=[120]))  
            
        weights1 = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1))  
        w_biases1 = tf.Variable(tf.zeros([84]))  
        weights2 = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1)) 
        w_biases2 = tf.Variable(tf.zeros([10]))  
            
        def model(data):
            # data (batch, 28, 28, 1)
            # filter1 (1, 1, 1, 6)
            conv = tf.nn.conv2d(data, filter1, [1,1,1,1], padding='SAME')
            conv = tf.nn.tanh(conv + biases1)
        # data reshaped to (batch, 28, 28, 1)
        # filter1 reshaped yo (1*1*1, 6)
        # conv shape (batch, 28, 28, 6)
        # sub-smapling
            conv = tf.nn.avg_pool(conv, [1,2,2,1], [1,2,2,1], padding='SAME')
        # conv shape(batch, 14, 14, 6)
        # filter2 shape(5, 5, 6, 16)
            conv = tf.nn.conv2d(conv, filter2, [1,1,1,1], padding='VALID')
        # conv reshaped to (batch, 10, 10, 5*5*6)
        # filter2 reshaped to (5*5*6, 16)
        # conv shape (batch, 10, 10, 16)
            conv = tf.nn.tanh(conv + biases2)
        # conv shape (batch, 10, 10, 16)
            conv = tf.nn.avg_pool(conv, [1,2,2,1], [1,2,2,1], padding='SAME')
        # conv shape (batch, 5,5 16)
        # filter3 shape (5,5, 16, 120)
            conv = tf.nn.conv2d(conv, filter3, [1,1,1,1], padding='VALID')
        # conv reshape( batch, 1, 1, 5*5*16)
        # filter3 reshape (5*5*16, 120)
        # conv = (batch, 1,1, 120)
            conv = tf.nn.tanh(conv + biases3)
            shape = conv.get_shape().as_list()
            reshape = tf.reshape(conv, (shape[0], reduce(lambda a,b:a*b, shape[1:])))
            hidden = tf.nn.relu(tf.matmul(reshape, weights1) + w_biases1) 
            hidden = tf.nn.dropout(hidden, 0.8)
            logits = tf.matmul(hidden, weights2) + w_biases2
            return logits

         # Training computation.  
        logits = model(tf_train_dataset)  
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))  
              
            # Optimizer.  
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  
            
            # Predictions for the training, validation, and test data.  
        train_prediction = tf.nn.softmax(logits)  
        test_prediction = tf.nn.softmax(model(tf_test_dataset))  



    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                
                print('Minibatch loss at step %d: %f' % (step, l))
                
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))  
    
def main():
    softmax_tf()
    
if __name__ == "__main__":
    import time

    # Beginning time
    program_begin_time = time.time()
    begin_ctime        = time.ctime()

    #  Run the main program
    main()

    # Print the time taken
    program_end_time = time.time()
    end_ctime        = time.ctime()
    seconds          = program_end_time - program_begin_time
    m, s             = divmod(seconds, 60)
    h, m             = divmod(m, 60)
    d, h             = divmod(h, 24)
    print("\n\nBegin time: ", begin_ctime)
    print("End   time: ", end_ctime, "\n")
    print("Time taken: {0: .0f} days, {1: .0f} hours, \
      {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s))
