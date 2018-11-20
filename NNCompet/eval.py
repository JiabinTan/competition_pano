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
log_file = open("message_valid.log", "w")
# redirect print output to log file
sys.stdout = log_file


path="/root/train/"
pd_path="./model/"
batch_size=5
filename=[path+'TFcodeX_'+str(10)+".tfrecord"]
filename_queue = tf.train.string_input_producer(filename,num_epochs=None) #读入流中
inputs, labels ,ids= decode_from_tfrecords(filename_queue, is_batch=False,batch_size=batch_size)
inputs=tf.clip_by_value((inputs/2+0.5)*255,0,255)
inputs=tf.cast(inputs,tf.uint8)
inputs=tf.expand_dims(inputs,3)
inputs=get_train_data(inputs,height=256,width=256,batch_size=batch_size)
labels=tf.subtract(labels,1)
onehots=tf.squeeze(tf.one_hot(labels,5,dtype=tf.int64),squeeze_dims=1)

inputs_aloc=tf.placeholder(
    dtype=tf.float32,
    shape=[None,256,256,1],
    name='input'
)
onehots_aloc=tf.placeholder(
    dtype=tf.int64,
    shape=[None,5],
    name='one_hot'
)
labels_aloc=tf.placeholder(
    dtype=tf.int64,
    shape=[None,1],
    name='label'
)

#global_steps = tf.Variable(1, trainable=False)

aux,prediction = inception_resnet_v2(inputs_aloc, num_classes=5, is_training=False,

                        reuse=None,

                        scope='InceptionResnetV2',

                        create_aux_logits=True,

                        activation_fn=tf.nn.leaky_relu)
loss = tf.losses.softmax_cross_entropy(
    onehots_aloc,
    aux,
    scope='Loss'
)

#tf.summary.scalar(
#    'Loss',
#    loss
#)

#train_step=tf.train.AdamOptimizer(0.0001).minimize(loss,global_step=global_steps)


    
accuraty=tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(prediction,dimension=1),tf.squeeze(labels_aloc)),tf.float32))
#tf.summary.scalar(
#    'Accuraty',
#    accuraty
#)

#merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
var_list = tf.trainable_variables()

g_list = tf.global_variables()

bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]

bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]

var_list += bn_moving_vars

saver = tf.train.Saver(var_list=var_list)

with tf.Session() as sess:
    sess.run(init)
    status=tf.train.latest_checkpoint(pd_path)
    if status:
        saver.restore(sess,status)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    ls=0.0
    acc=0.0
    for i in range(70):
        print('========>',i/70*100,'%')
        inputs_,labels_,onehots_=sess.run([inputs,labels,onehots])
        loss_,accuraty_=sess.run([loss,accuraty],\
            feed_dict={inputs_aloc:inputs_,labels_aloc:labels_,onehots_aloc:onehots_})
        ls+=loss_;
        acc+=accuraty_;

    print('accuracy  is ',acc/70)
    print('loss  is ',ls/70)  
    print('train done')        
    sys.stdout.flush()
    coord.request_stop()
    coord.join(threads)
