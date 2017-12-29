import io
import sys
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2

def ts1_res3d(inputs, weight_decay):
  # Res3D Block 1
  conv3d_1 = keras.layers.Conv3D(64, (3,7,7), strides=(1,2,2), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_1')(inputs)
  conv3d_1 = keras.layers.BatchNormalization(name='BatchNorm_1_0')(conv3d_1)
  conv3d_1 = keras.layers.Activation('relu', name='ReLU_1')(conv3d_1)
 
  # Res3D Block 2
  conv3d_2a_1 = keras.layers.Conv3D(64, (1,1,1), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2a_1')(conv3d_1)
  conv3d_2a_1 = keras.layers.BatchNormalization(name='BatchNorm_2a_1')(conv3d_2a_1)
  conv3d_2a_a = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2a_a')(conv3d_1)
  conv3d_2a_a = keras.layers.BatchNormalization(name='BatchNorm_2a_a')(conv3d_2a_a)
  conv3d_2a_a = keras.layers.Activation('relu', name='ReLU_2a_a')(conv3d_2a_a)
  conv3d_2a_b = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2a_b')(conv3d_2a_a)
  conv3d_2a_b = keras.layers.BatchNormalization(name='BatchNorm_2a_b')(conv3d_2a_b)
  conv3d_2a = keras.layers.Add(name='Add_2a')([conv3d_2a_1, conv3d_2a_b])
  conv3d_2a = keras.layers.Activation('relu', name='ReLU_2a')(conv3d_2a)

  conv3d_2b_a = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2b_a')(conv3d_2a)
  conv3d_2b_a = keras.layers.BatchNormalization(name='BatchNorm_2b_a')(conv3d_2b_a)
  conv3d_2b_a = keras.layers.Activation('relu', name='ReLU_2b_a')(conv3d_2b_a)
  conv3d_2b_b = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2b_b')(conv3d_2b_a)
  conv3d_2b_b = keras.layers.BatchNormalization(name='BatchNorm_2b_b')(conv3d_2b_b)
  conv3d_2b = keras.layers.Add(name='Add_2b')([conv3d_2a, conv3d_2b_b])
  conv3d_2b = keras.layers.Activation('relu', name='ReLU_2b')(conv3d_2b)
  conv3d_2b = keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2), 
                    padding='same', name='Conv3d_2b_Pooling')(conv3d_2b)

  # Res3D Block 3
  conv3d_3a_1 = keras.layers.Conv3D(128, (1,1,1), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3a_1')(conv3d_2b)
  conv3d_3a_1 = keras.layers.BatchNormalization(name='BatchNorm_3a_1')(conv3d_3a_1)
  conv3d_3a_a = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3a_a')(conv3d_2b)
  conv3d_3a_a = keras.layers.BatchNormalization(name='BatchNorm_3a_a')(conv3d_3a_a)
  conv3d_3a_a = keras.layers.Activation('relu', name='ReLU_3a_a')(conv3d_3a_a)
  conv3d_3a_b = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3a_b')(conv3d_3a_a)
  conv3d_3a_b = keras.layers.BatchNormalization(name='BatchNorm_3a_b')(conv3d_3a_b)
  conv3d_3a = keras.layers.Add(name='Add_3a')([conv3d_3a_1, conv3d_3a_b])
  conv3d_3a = keras.layers.Activation('relu', name='ReLU_3a')(conv3d_3a)

  conv3d_3b_a = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3b_a')(conv3d_3a)
  conv3d_3b_a = keras.layers.BatchNormalization(name='BatchNorm_3b_a')(conv3d_3b_a)
  conv3d_3b_a = keras.layers.Activation('relu', name='ReLU_3b_a')(conv3d_3b_a)
  conv3d_3b_b = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3b_b')(conv3d_3b_a)
  conv3d_3b_b = keras.layers.BatchNormalization(name='BatchNorm_3b_b')(conv3d_3b_b)
  conv3d_3b = keras.layers.Add(name='Add_3b')([conv3d_3a, conv3d_3b_b])
  conv3d_3b = keras.layers.Activation('relu', name='ReLU_3b')(conv3d_3b)
  conv3d_3b = keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2), 
                    padding='same', name='Conv3d_3b_Pooling')(conv3d_3b)

  # Res3D Block 4
  conv3d_4a_1 = keras.layers.Conv3D(256, (1,1,1), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4a_1')(conv3d_3b)
  conv3d_4a_1 = keras.layers.BatchNormalization(name='BatchNorm_4a_1')(conv3d_4a_1)
  conv3d_4a_a = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4a_a')(conv3d_3b)
  conv3d_4a_a = keras.layers.BatchNormalization(name='BatchNorm_4a_a')(conv3d_4a_a)
  conv3d_4a_a = keras.layers.Activation('relu', name='ReLU_4a_a')(conv3d_4a_a)
  conv3d_4a_b = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4a_b')(conv3d_4a_a)
  conv3d_4a_b = keras.layers.BatchNormalization(name='BatchNorm_4a_b')(conv3d_4a_b)
  conv3d_4a = keras.layers.Add(name='Add_4a')([conv3d_4a_1, conv3d_4a_b])
  conv3d_4a = keras.layers.Activation('relu', name='ReLU_4a')(conv3d_4a)

  conv3d_4b_a = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4b_a')(conv3d_4a)
  conv3d_4b_a = keras.layers.BatchNormalization(name='BatchNorm_4b_a')(conv3d_4b_a)
  conv3d_4b_a = keras.layers.Activation('relu', name='ReLU_4b_a')(conv3d_4b_a)
  conv3d_4b_b = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4b_b')(conv3d_4b_a)
  conv3d_4b_b = keras.layers.BatchNormalization(name='BatchNorm_4b_b')(conv3d_4b_b)
  conv3d_4b = keras.layers.Add(name='Add_4b')([conv3d_4a, conv3d_4b_b])
  conv3d_4b = keras.layers.Activation('relu', name='ReLU_4b')(conv3d_4b)

  # Res3D Block 5
  conv3d_5a_1 = keras.layers.Conv3D(512, (1,1,1), strides=(1,2,2), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_5a_1')(conv3d_4b)
  conv3d_5a_1 = keras.layers.BatchNormalization(name='BatchNorm_5a_1')(conv3d_5a_1)
  conv3d_5a_a = keras.layers.Conv3D(512, (3,3,3), strides=(1,2,2), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_5a_a')(conv3d_4b)
  conv3d_5a_a = keras.layers.BatchNormalization(name='BatchNorm_5a_a')(conv3d_5a_a)
  conv3d_5a_a = keras.layers.Activation('relu', name='ReLU_5a_a')(conv3d_5a_a)
  conv3d_5a_b = keras.layers.Conv3D(512, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_5a_b')(conv3d_5a_a)
  conv3d_5a_b = keras.layers.BatchNormalization(name='BatchNorm_5a_b')(conv3d_5a_b)
  conv3d_5a = keras.layers.Add(name='Add_5a')([conv3d_5a_1, conv3d_5a_b])
  conv3d_5a = keras.layers.Activation('relu', name='ReLU_5a')(conv3d_5a)

  conv3d_5b_a = keras.layers.Conv3D(512, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_5b_a')(conv3d_5a)
  conv3d_5b_a = keras.layers.BatchNormalization(name='BatchNorm_5b_a')(conv3d_5b_a)
  conv3d_5b_a = keras.layers.Activation('relu', name='ReLU_5b_a')(conv3d_5b_a)
  conv3d_5b_b = keras.layers.Conv3D(512, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_5b_b')(conv3d_5b_a)
  conv3d_5b_b = keras.layers.BatchNormalization(name='BatchNorm_5b_b')(conv3d_5b_b)
  conv3d_5b = keras.layers.Add(name='Add_5b')([conv3d_5a, conv3d_5b_b])
  conv3d_5b = keras.layers.Activation('relu', name='ReLU_5b')(conv3d_5b)

  gpooling = keras.layers.AveragePooling3D(pool_size=(1,7,7), strides=(1,7,7), padding='same',
                    name='Average_Pooling')(conv3d_5b)
  gpooling = keras.layers.Reshape((512,))(gpooling)
  return gpooling

