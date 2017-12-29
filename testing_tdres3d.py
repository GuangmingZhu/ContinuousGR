import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import io
import sys
sys.path.append("./networks")
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
K=tf.contrib.keras.backend
import inputs as data
from temporal_dilated_res3d import td_res3d
from callbacks import LearningRateScheduler 
from datagen import conTrainImageBoundaryGenerator, conTestImageBoundaryGenerator
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime

RGB = 0
Depth = 1
depth = 128
batch_size = 1
num_classes = 1
weight_decay = 0.00005
model_prefix = '.'

inputs = keras.layers.Input(shape=(None, 112, 112, 3))
feature = td_res3d(inputs, weight_decay)
prob = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), name='Classes_Bound')(feature)
outputs = keras.layers.Activation('sigmoid', name='Output')(prob)
model = keras.models.Model(inputs=inputs, outputs=outputs)
optimizer = keras.optimizers.SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='balanced_squared_hinge', metrics=['accuracy'])

########################################################################################
########################################################################################
pretrained_model = '%s/trained_models/tdres3d/congr_tdres3d_rgb_weights.h5'%model_prefix
print 'Loading pretrained model from %s' % pretrained_model
model.load_weights(pretrained_model, by_name=False)
for i in range(len(model.trainable_weights)):
  print model.trainable_weights[i]

testing_datalist = './dataset_splits/ConGD/valid_rgb_list.txt'
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
rgb_prediction = model.predict_generator(conTestImageBoundaryGenerator(testing_datalist, 
                                                                         batch_size, depth, num_classes, RGB),
                                           steps=test_steps,
                                          )
np.save('features/rgb_valid_boundscore_tdres3d.npy', rgb_prediction)

testing_datalist = './dataset_splits/ConGD/test_rgb_list.txt'
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
rgb_prediction = model.predict_generator(conTestImageBoundaryGenerator(testing_datalist, 
                                                                         batch_size, depth, num_classes, RGB),
                                           steps=test_steps,
                                          )
np.save('features/rgb_test_boundscore_tdres3d.npy', rgb_prediction)

########################################################################################
########################################################################################
pretrained_model = '%s/trained_models/tdres3d/congr_tdres3d_depth_weights.h5'%model_prefix
print 'Loading pretrained model from %s' % pretrained_model
model.load_weights(pretrained_model, by_name=False)
for i in range(len(model.trainable_weights)):
  print model.trainable_weights[i]

testing_datalist = './dataset_splits/ConGD/valid_depth_list.txt'
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
depth_prediction = model.predict_generator(conTestImageBoundaryGenerator(testing_datalist, 
                                                                         batch_size, depth, num_classes, Depth),
                                           steps=test_steps,
                                          )
np.save('features/depth_valid_boundscore_tdres3d.npy', depth_prediction)

testing_datalist = './dataset_splits/ConGD/test_depth_list.txt'
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
depth_prediction = model.predict_generator(conTestImageBoundaryGenerator(testing_datalist, 
                                                                         batch_size, depth, num_classes, Depth),
                                           steps=test_steps,
                                          )
np.save('features/depth_test_boundscore_tdres3d.npy', depth_prediction)
