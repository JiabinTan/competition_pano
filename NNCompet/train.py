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

global_steps = tf.Variable(1, trainable=False,name='global_step')

aux,prediction = inception_resnet_v2(inputs_aloc, num_classes=5, is_training=True,

                        reuse=None,

                        scope='InceptionResnetV2',

                        create_aux_logits=True,

                        activation_fn=tf.nn.leaky_relu)
loss = tf.losses.softmax_cross_entropy(
    onehots_aloc,
    aux,
    scope='Loss'
)
W=tf.get_default_graph().get_tensor_by_name("InceptionResnetV2/Conv2d_1a_3x3/weights:0")
B=tf.get_default_graph().get_tensor_by_name("InceptionResnetV2/Conv2d_1a_3x3/biases:0")
d_1=tf.gradients(loss,[W,B]) 
W=tf.get_default_graph().get_tensor_by_name("InceptionResnetV2/AuxLogits/Conv2d_2a_5x5/weights:0")
B=tf.get_default_graph().get_tensor_by_name("InceptionResnetV2/AuxLogits/Conv2d_2a_5x5/biases:0")
d_2=tf.gradients(loss,[W,B]) 
tf.summary.scalar(
    'Loss',
    loss
)

#train_step=tf.train.MomentumOptimizer(1e-6,0.9,use_nestrov=True).minimize(loss,global_step=global_steps)

train_step=tf.train.AdamOptimizer(learning_rate=1e-8).minimize(loss,global_step=global_steps)

    
accuraty=tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(prediction,dimension=1),tf.squeeze(labels_aloc)),tf.float32))
tf.summary.scalar(
    'Accuraty',
    accuraty
)

merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
var_list = tf.trainable_variables()

g_list = tf.global_variables()

bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]

bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]

var_list += bn_moving_vars

var_list +=[global_steps]

saver = tf.train.Saver(var_list=var_list)
sess.graph.finalize()
with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_path, sess.graph)
    sess.run(init)
    status=tf.train.latest_checkpoint(pd_path)
    if status:
        saver.restore(sess,status)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    ls=0.0
    acc=0.0
    can_index=[0]
    can_ls=[0]
    can_acc=[0]
    epoch=20000
    for i in range(epoch):
        time_start=time.time()
        inputs_,labels_,onehots_=sess.run([inputs,labels,onehots])
        loss_,aux_,prediction_,_,merged_,accuraty_,d_1_,d_2_=sess.run([loss,aux,prediction,train_step,merged,accuraty,d_1,d_2],\
            feed_dict={inputs_aloc:inputs_,labels_aloc:labels_,onehots_aloc:onehots_})
        writer.add_summary(merged_,i)
        ls+=loss_;
        acc+=accuraty_;
        time_end=time.time()
        if i%50==0:
            print('<========epoch : ',i,'========>')
            print('using time : ',time_end-time_start)
            print('grad : ',np.max(d_1_[0]),'<====>',np.min(d_1_[0]))
            print('grad : ',np.max(d_1_[1]),'<====>',np.min(d_1_[1]))
            print('grad : ',np.max(d_2_[0]),'<====>',np.min(d_2_[0]))
            print('grad : ',np.max(d_2_[1]),'<====>',np.min(d_2_[1]))
            print('aux ；',aux_)
            print('prediction: ',prediction_)
            can_index+=[i]
            print('accuracy  is ',acc/50)
            can_acc+=[acc/50];
            print('loss  is ',ls/50)
            can_ls+=[ls/50]
            
            ls=0.0
            acc=0.0
            sys.stdout.flush()
        if (i%1000==0)&(i!=0)&(i!=epoch-1):
            saver.save(sess,pd_path+'net.ckpt',global_step=global_steps)
    print('train done')
    saver.save(sess, pd_path+'net.ckpt', global_step=global_steps)
    plt.plot(can_index,can_acc,'b',label='acc')
    plt.plot(can_index,can_ls,'r',label='loss')
    plt.legend()
    plt.savefig(fig_save_path+'ls_acc.pdf')
            
    coord.request_stop()
    coord.join(threads)