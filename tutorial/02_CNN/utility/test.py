import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import help_function as h

import cifar10
cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()
class_names
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()


from cifar10 import img_size, num_channels, num_classes
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


size = 5
channels = 3 #shape of input image: [32,32,3]
filters = 64
stride = 1
conv1_w = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
conv1_b = tf.Variable(tf.truncated_normal([filters], stddev=0.1))    
net = tf.nn.conv2d(x, conv1_w, strides=[1, stride, stride, 1], padding='SAME',name='conv1')
net = tf.add(net,conv1_b)
net = tf.nn.relu(net)

size = 2
stride = 2
net = tf.nn.max_pool(net, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME', name='pooling1')


k_size = 5
channels = 64 
filters = 128
stride = 1
conv2_w = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
conv2_b = tf.Variable(tf.truncated_normal([filters], stddev=0.1))    
net = tf.nn.conv2d(net, conv2_w, strides=[1, stride, stride, 1], padding='SAME',name='conv2')
net = tf.add(net,conv2_b)
net = tf.nn.relu(net)

size = 2
stride = 2
net = tf.nn.max_pool(net, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME', name='pooling2')

#Flatten 
net_shape = net.get_shape().as_list()
dim = net_shape[1]*net_shape[2]*net_shape[3]
net_transposed = tf.transpose(net,(0,3,1,2))
net = tf.reshape(net_transposed, [-1,dim])

fc1_w = tf.Variable(tf.truncated_normal([dim,256], stddev=0.1))
fc1_b = tf.Variable(tf.truncated_normal([256], stddev=0.1)) 
net = tf.add(tf.matmul(net,fc1_w),fc1_b)
net = tf.nn.relu(net)

fc2_w = tf.Variable(tf.truncated_normal([256,128], stddev=0.1))
fc2_b = tf.Variable(tf.truncated_normal([128], stddev=0.1)) 
net = tf.add(tf.matmul(net,fc2_w),fc2_b)

fc3_w = tf.Variable(tf.truncated_normal([128,10], stddev=0.1))
fc3_b = tf.Variable(tf.truncated_normal([10], stddev=0.1)) 
logits = tf.add(tf.matmul(net,fc3_w),fc3_b)

y_pred = tf.nn.softmax(logits)
y_pred_label = tf.arg_max(tf.nn.softmax(y_pred),dimension=1)


#============Loss================



loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_true) 
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_batch, y_true_batch = h.random_batch(images_train, labels_train, 32)


    feed_dict_train = {x: x_batch,
                       y_true: y_true_batch}
    
    a = sess.run([y_pred_label, y_pred], feed_dict=feed_dict_train)
    
    

