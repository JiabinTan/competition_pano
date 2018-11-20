# -*- coding: UTF-8 -*-

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function





import tensorflow as tf
from tensorflow.contrib import slim







def block35(net, scale=1.0, activation_fn=tf.nn.leaky_relu, scope=None, reuse=None):

  """Builds the 35x35 resnet block."""

  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):

    with tf.variable_scope('Branch_0'):

      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')

    with tf.variable_scope('Branch_1'):

      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')

      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')

    with tf.variable_scope('Branch_2'):

      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')

      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')

      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')

    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])

    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,

                     activation_fn=None, scope='Conv2d_1x1')

    scaled_up = up * scale

    if activation_fn == tf.nn.relu6:

      # Use clip_by_value to simulate bandpass activation.

      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)



    net += scaled_up

    if activation_fn:

      net = activation_fn(net)

  return net





def block17(net, scale=1.0, activation_fn=tf.nn.leaky_relu, scope=None, reuse=None):

  """Builds the 17x17 resnet block."""

  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):

    with tf.variable_scope('Branch_0'):

      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')

    with tf.variable_scope('Branch_1'):

      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')

      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],

                                  scope='Conv2d_0b_1x7')

      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],

                                  scope='Conv2d_0c_7x1')

    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])

    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,

                     activation_fn=None, scope='Conv2d_1x1')



    scaled_up = up * scale

    if activation_fn == tf.nn.relu6:

      # Use clip_by_value to simulate bandpass activation.

      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)



    net += scaled_up

    if activation_fn:

      net = activation_fn(net)

  return net







def inception_resnet_v2_base(inputs,

                             final_endpoint='PreAuxLogits',

                             output_stride=16,

                             align_feature_maps=False,

                             scope=None,

                             activation_fn=tf.nn.leaky_relu):


  if output_stride != 8 and output_stride != 16:

    raise ValueError('output_stride must be 8 or 16.')



  padding = 'SAME' if align_feature_maps else 'VALID'






  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):

    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],

                        stride=1, padding='SAME'):

      # 149 x 149 x 32

      net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding,

                        scope='Conv2d_1a_3x3')



      


      # 147 x 147 x 32

      net = slim.conv2d(net, 32, 3, padding=padding,

                        scope='Conv2d_2a_3x3')


      # 147 x 147 x 64

      net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')


      # 73 x 73 x 64

      net = slim.max_pool2d(net, 3, stride=2, padding=padding,

                            scope='MaxPool_3a_3x3')


      # 73 x 73 x 80

      net = slim.conv2d(net, 80, 1, padding=padding,

                        scope='Conv2d_3b_1x1')


      # 71 x 71 x 192

      net = slim.conv2d(net, 192, 3, padding=padding,

                        scope='Conv2d_4a_3x3')

      # 35 x 35 x 192

      net = slim.max_pool2d(net, 3, stride=2, padding=padding,

                            scope='MaxPool_5a_3x3')



      # 35 x 35 x 320

      with tf.variable_scope('Mixed_5b'):

        with tf.variable_scope('Branch_0'):

          tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')

        with tf.variable_scope('Branch_1'):

          tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')

          tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,

                                      scope='Conv2d_0b_5x5')

        with tf.variable_scope('Branch_2'):

          tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')

          tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,

                                      scope='Conv2d_0b_3x3')

          tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,

                                      scope='Conv2d_0c_3x3')

        with tf.variable_scope('Branch_3'):

          tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',

                                       scope='AvgPool_0a_3x3')

          tower_pool_1 = slim.conv2d(tower_pool, 64, 1,

                                     scope='Conv2d_0b_1x1')

        net = tf.concat(

            [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)




      # TODO(alemi): Register intermediate endpoints

      net = slim.repeat(net, 10, block35, scale=0.17,

                        activation_fn=activation_fn)



      # 17 x 17 x 1088 if output_stride == 8,

      # 33 x 33 x 1088 if output_stride == 16

      use_atrous = output_stride == 8



      with tf.variable_scope('Mixed_6a'):

        with tf.variable_scope('Branch_0'):

          tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,

                                   padding=padding,

                                   scope='Conv2d_1a_3x3')

        with tf.variable_scope('Branch_1'):

          tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')

          tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,

                                      scope='Conv2d_0b_3x3')

          tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,

                                      stride=1 if use_atrous else 2,

                                      padding=padding,

                                      scope='Conv2d_1a_3x3')

        with tf.variable_scope('Branch_2'):

          tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,

                                       padding=padding,

                                       scope='MaxPool_1a_3x3')

        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)





      # TODO(alemi): register intermediate endpoints

      with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):

        net = slim.repeat(net, 20, block17, scale=0.10,

                          activation_fn=activation_fn)
        return net



def inception_resnet_v2(inputs, num_classes=1001, is_training=True,

                        reuse=None,

                        scope='InceptionResnetV2',

                        create_aux_logits=True,

                        activation_fn=tf.nn.leaky_relu):

 


  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs],

                         reuse=reuse) as scope:

    with slim.arg_scope([slim.batch_norm, slim.dropout],

                        is_training=is_training):



      net = inception_resnet_v2_base(inputs, scope=scope,

                                                 activation_fn=activation_fn)



      if create_aux_logits and num_classes:

        with tf.variable_scope('AuxLogits'):

            
            net = slim.avg_pool2d(net, 5, stride=3, padding='VALID',
            
                                scope='Conv2d_1a_3x3')
            
            net = slim.conv2d(net, 128, 1, scope='Conv2d_1b_1x1')
            
            net = slim.conv2d(net, 768, net.get_shape()[1:3],
            
                            padding='VALID', scope='Conv2d_2a_5x5')
            
            net = slim.flatten(net)
            
            net = slim.fully_connected(net, num_classes, activation_fn=None,
            
                                    scope='Logits')
            
            prediction = tf.nn.softmax(net, name='Predictions')

            
    return net, prediction





def inception_resnet_v2_arg_scope(

    weight_decay=0.00004,

    batch_norm_decay=0.9997,

    batch_norm_epsilon=0.001,

    activation_fn=tf.nn.leaky_relu,

    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):

  """Returns the scope with the default parameters for inception_resnet_v2.



  Args:

    weight_decay: the weight decay for weights variables.

    batch_norm_decay: decay for the moving average of batch_norm momentums.

    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

    activation_fn: Activation function for conv2d.

    batch_norm_updates_collections: Collection for the update ops for

      batch norm.



  Returns:

    a arg_scope with the parameters needed for inception_resnet_v2.

  """

  # Set weight_decay for weights in conv2d and fully_connected layers.

  with slim.arg_scope([slim.conv2d, slim.fully_connected],

                      weights_regularizer=slim.l2_regularizer(weight_decay),

                      biases_regularizer=slim.l2_regularizer(weight_decay)):



    batch_norm_params = {

        'decay': batch_norm_decay,

        'epsilon': batch_norm_epsilon,

        'updates_collections': batch_norm_updates_collections,

        'fused': None,  # Use fused batch norm if possible.

    }

    # Set activation_fn and parameters for batch_norm.

    with slim.arg_scope([slim.conv2d], activation_fn=activation_fn,

                        normalizer_fn=slim.batch_norm,

                        normalizer_params=batch_norm_params) as scope:

      return scope

