import numpy as np
import random
import tensorflow as tf

def ramdom_batch(data_length, batch_size):
    fid_list = list(range(data_length))
    random.shuffle(fid_list) 
    datapool = []
        
    for i in range(data_length//batch_size):
        datapool.append(fid_list[i*batch_size:i*batch_size + batch_size])
        
        
    return datapool


def print_operations_in_graph():
    
    graph = tf.get_default_graph()
    
    for op in graph.get_operations(): print(op.name)
    
    
def clac_accuracy(logits, labels):
    
    with tf.name_scope('accuracy'):
        predict_model = tf.argmax(tf.nn.softmax(logits),1)
        true_label = tf.argmax(labels,1)   
        true_positive = tf.reduce_sum(tf.cast(tf.equal(predict_model, true_label), tf.int32))
    
    return true_positive