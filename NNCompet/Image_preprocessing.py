import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

'''
有可能文件没有保存，只需要设置下返回值就好
'''

#参数说明
#依次是输入文件数量，文件名前缀

    
#参数说明：
#保存的文件名，image 数组，label数组，id数组。
#id处理随便，甚至去常量都可以，反正训练的时候不需要
def encode_to_tfrecords(tfrecords_filename,images,labels,ids): 
    ''' write into tfrecord file '''
    if os.path.exists(tfrecords_filename):
        os.remove(tfrecords_filename)
 
    writer = tf.python_io.TFRecordWriter('./'+tfrecords_filename) # 创建.tfrecord文件，准备写入

    example = tf.train.Example(features=tf.train.Features(
            feature={
            'data': tf.train.Feature(float_list = tf.train.FloatList(value=images)),     
            'label':tf.train.Feature(int64_list = tf.train.Int64List(value = labels)),
            'id':tf.train.Feature(int64_list = tf.train.Int64List(value = ids))
            }))
    writer.write(example.SerializeToString()) 
 
    writer.close()
    print(tfrecords_filename+"保存成功！") 
    return 0;
    
#参数说明：
#分别是文件名序列，是否使用batch，后面都是shuffle_batch函数中的参量具体看官方文档
#
def decode_from_tfrecords(filename_queue, is_batch,batch_size = 128,num_threads=2,capacity=500,min_after_dequeue=100):
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                        features={
                                            'data': tf.FixedLenFeature([256,256], tf.float32),
                                            'id' : tf.FixedLenFeature([1], tf.int64),
                                            'label' : tf.FixedLenFeature([1], tf.int64)
                                        })  #取出包含image和label的feature对象
    #image = tf.decode_raw(features['img_raw'],tf.int64)
    #image = tf.reshape(image, [56,56])
        
    image = features['data']
    label = features['label']
    id=features['id']
    if is_batch:
        image, label ,id= tf.train.shuffle_batch([image, label,id],
                                                            batch_size=batch_size, 
                                                            num_threads=num_threads, 
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
    return image, label,id
 
        
 
#    run_test = True
train_filename=[]
for i in range(1):
    train_filename=["G:\\迅雷下载\\大象分形杯人工智能挑战赛\\大象分形杯人工智能挑战赛\\TFcodeX_"+str(i+1)+".tfrecord"]

    filename_queue = tf.train.string_input_producer(train_filename,num_epochs=None) #读入流中
    raw_image, raw_label ,raw_id= decode_from_tfrecords(filename_queue, is_batch=True)
    ##接下来这一个部分是你进行数据处理的部分，也就是对这三个变量进行处理raw_image, raw_label ,raw_id
    ##处理后变量名确定为 prod_image,prod_label,prod_id
    ##单文件批处理，每读取一个文件，提取文件中图片，并且每次提取128张图片，具体提取数量可以通过设置decode_from_tfrecords中batch_size来设置
    ##处理的图片是数组形式（list）
    ##你的代码

    ##
    ##
        

    
with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    a='1'
    try:
        # while not coord.should_stop():
        print("每次调用得到的图片数量：")
        for i in range(10):
            example ,l= sess.run([raw_image,raw_label])#在会话中取出image和label
            print(len(example))
            #plt.subplot(2,5,i+1)
            #plt.imshow(example) # 显示图片
            pass
        #plt.show()
        #save_filename='';
        #encode_to_tfrecords(save_filename,prod_image,prod_label,prod_id)
    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop()
 
    coord.request_stop()
    coord.join(threads)