# -*- coding: UTF-8 -*-
import tensorflow as tf
from model import inception_resnet_v2
from reader import decode_from_tfrecords
from reader import get_train_data
import time
import sys
import matplotlib.pyplot as plt 
import numpy as np

# make a copy of original stdout route
stdout_backup = sys.stdout
# define the log file that receives your log info 
log_file = open("message.log", "w")
# redirect print output to log file
sys.stdout = log_file


path="/root/train/"
log_path =  "./tf_writer"
pd_path="./model/"
fig_save_path='./fig/'
train_filename=[]
batch_size=2
for j in range(9):
    filename=[path+'TFcodeX_'+str(j+1)+".tfrecord"]
    train_filename+=filename
filename_queue = tf.train.string_input_producer(train_filename,num_epochs=None) #读入流中
inputs, labels ,ids= decode_from_tfrecords(filename_queue, is_batch=True,batch_size=batch_size)
inputs=tf.clip_by_value((inputs/2+0.5)*255,0,255)
inputs=tf.cast(inputs,tf.uint8)
inputs=tf.expand_dims(inputs,3)
inputs=get_train_data(inputs,height=256,width=256,batch_size=batch_size)



init = tf.global_variables_initializer()




with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(10):
        
        inputs_=sess.run([inputs])
        plt.imsave(str(i)+'0.jpg',inputs_[0])
        plt.imsave(str(i)+'1.jpg',inputs_[1])
            
    coord.request_stop()
    coord.join(threads)