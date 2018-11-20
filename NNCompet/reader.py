# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import control_flow_ops
#使用哪种方式来缩放图片
def apply_with_random_selector(x, func, num_cases):
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

#图片亮度、对比度等操作
#此处因为输入的tensor为[:,:,1]
def distort_color(image, color_ordering=0, scope=None):
    """
    must be [:,:,:,1] 
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            #image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            #image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            #image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            #image = tf.image.random_hue(image, max_delta=0.2)
            #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            raise ValueError('color_ordering must be in [0, 3]')
        pass

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)

 
def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """
    截取部分图片
    image=[:,:,:,1]
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of
        # interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the
        # human-annotated
        # bounding box.  If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox


def preprocess_for_train(image, height, width, bbox,
                         scope=None,
                         add_image_summaries=True):

  with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
    if bbox is None:
      bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)


    distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox,aspect_ratio_range=[0.8,1.2],
                                                                  min_object_covered=0.75)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None,1])



    num_resize_cases = 1 
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=num_resize_cases)



    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors. There are 1 or 4 ways to do it.
    num_distort_cases = 1 
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering),
        num_cases=num_distort_cases)

    #distorted_image = tf.subtract(distorted_image, 0.5)
    #distorted_image = tf.multiply(distorted_image, 2.0)
    return distorted_image


def get_train_data(x,height=299,width=299,batch_size=34):

    x=tf.stack([preprocess_for_train(i,height,width,None,add_image_summaries=True)
                            for i in tf.unstack(x,num=batch_size,axis=0)])
    return x



def decode_from_tfrecords(filename_queue, is_batch,batch_size = 128,num_threads=5,capacity=1000,min_after_dequeue=500):
    """
    return form [batch,wdith,height,tunnal]
    """
    reader_in = tf.TFRecordReader()
    _, serialized_example = reader_in.read(filename_queue)   #返回文件名和文件
    #num=reader.num_records_produced()
    features = tf.parse_single_example(serialized_example,
                                        features={
                                            'data': tf.FixedLenFeature([256,256], tf.float32),
                                            'label' : tf.FixedLenFeature([1], tf.int64),
                                            'id' : tf.FixedLenFeature([1], tf.int64)
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

    else :
        image, label ,id=tf.train.batch([image, label,id],
                                                    batch_size=batch_size, 
                                                    num_threads=num_threads, 
                                                    capacity=capacity)

    return image, label,id