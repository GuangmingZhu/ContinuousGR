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
from res3d_clstm_mobilenet import res3d_clstm_mobilenet
from callbacks import LearningRateScheduler 
from datagen import conTrainImageGenerator, conTestImageGenerator
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime

RGB = 0
Depth = 1
Flow = 2
depth = 32
batch_size = 1
num_classes = 249
weight_decay = 0.00005
model_prefix = '.'
  
inputs = keras.layers.Input(shape=(depth, 112, 112, 3),
                            batch_shape=(batch_size, depth, 112, 112, 3))
feature = res3d_clstm_mobilenet(inputs, depth, weight_decay)
model = keras.models.Model(inputs=inputs, outputs=feature)
optimizer = keras.optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

########################################################################################
########################################################################################
pretrained_model = '%s/trained_models/rcm/congr_rcm_rgb_weights.h5'%model_prefix
print 'Loading pretrained model from %s' % pretrained_model
model.load_weights(pretrained_model, by_name=True)
for i in range(len(model.trainable_weights)):
  print model.trainable_weights[i]

training_datalist = './dataset_splits/ConGD/train_rgb_isolist.txt'
train_data = data.load_con_video_list(training_datalist)
train_steps = len(train_data)/batch_size
rgb_trfeat = model.predict_generator(conTestImageGenerator(training_datalist, 
                                     batch_size, depth, num_classes, RGB),
                                     steps=train_steps,
                                     )
np.save('features/con_rgb_trfeat.npy', rgb_trfeat)

testing_datalist = './features/tdres3d/valid_rgb_predlist+.txt'
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
rgb_tefeat = model.predict_generator(conTestImageGenerator(testing_datalist, 
                                     batch_size, depth, num_classes, RGB),
                                     steps=test_steps,
                                     )
np.save('features/con_rgb_pvafeat+.npy', rgb_tefeat)

testing_datalist = './features/tdres3d/test_rgb_predlist+.txt'
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
rgb_tefeat = model.predict_generator(conTestImageGenerator(testing_datalist, 
                                     batch_size, depth, num_classes, RGB),
                                     steps=test_steps,
                                     )
np.save('features/con_rgb_ptefeat+.npy', rgb_tefeat)

########################################################################################
########################################################################################
pretrained_model = '%s/trained_models/rcm/congr_rcm_depth_weights.h5'%model_prefix
print 'Loading pretrained model from %s' % pretrained_model
model.load_weights(pretrained_model, by_name=True)
for i in range(len(model.trainable_weights)):
  print model.trainable_weights[i]

training_datalist = './dataset_splits/ConGD/train_depth_isolist.txt'
train_data = data.load_con_video_list(training_datalist)
train_steps = len(train_data)/batch_size
depth_trfeat = model.predict_generator(conTestImageGenerator(training_datalist, 
                                     batch_size, depth, num_classes, Depth),
                                     steps=train_steps,
                                     )
np.save('features/con_depth_trfeat.npy', depth_trfeat)

testing_datalist = './features/tdres3d/valid_depth_predlist+.txt'
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
depth_tefeat = model.predict_generator(conTestImageGenerator(testing_datalist, 
                                     batch_size, depth, num_classes, Depth),
                                     steps=test_steps,
                                     )
np.save('features/con_depth_pvafeat+.npy', depth_tefeat)

testing_datalist = './features/tdres3d/test_depth_predlist+.txt'
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
depth_tefeat = model.predict_generator(conTestImageGenerator(testing_datalist, 
                                     batch_size, depth, num_classes, Depth),
                                     steps=test_steps,
                                     )
np.save('features/con_depth_ptefeat+.npy', depth_tefeat)

########################################################################################
########################################################################################
pretrained_model = '%s/trained_models/rcm/congr_rcm_flow_weights.h5'%model_prefix
print 'Loading pretrained model from %s' % pretrained_model
model.load_weights(pretrained_model, by_name=True)
for i in range(len(model.trainable_weights)):
  print model.trainable_weights[i]

training_datalist = './dataset_splits/ConGD/train_flow_isolist.txt'
train_data = data.load_con_video_list(training_datalist)
train_steps = len(train_data)/batch_size
flow_trfeat = model.predict_generator(conTestImageGenerator(training_datalist, 
                                     batch_size, depth, num_classes, Flow),
                                     steps=train_steps,
                                     )
np.save('features/con_flow_trfeat.npy', flow_trfeat)

testing_datalist = './features/tdres3d/valid_flow_predlist+.txt'
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
flow_tefeat = model.predict_generator(conTestImageGenerator(testing_datalist, 
                                     batch_size, depth, num_classes, Flow),
                                     steps=test_steps,
                                     )
np.save('features/con_flow_pvafeat+.npy', flow_tefeat)

testing_datalist = './features/tdres3d/test_flow_predlist+.txt'
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
flow_tefeat = model.predict_generator(conTestImageGenerator(testing_datalist, 
                                     batch_size, depth, num_classes, Flow),
                                     steps=test_steps,
                                     )
np.save('features/con_flow_ptefeat+.npy', flow_tefeat)

