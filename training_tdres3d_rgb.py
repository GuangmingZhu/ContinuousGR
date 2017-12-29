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
nb_epoch = 10
init_epoch = 0
depth = 128
batch_size = 1
num_classes = 1
weight_decay = 0.00005
dataset_name = 'congr_tdres3d_rgb'
training_datalist = './dataset_splits/ConGD/train_rgb_list.txt'
testing_datalist = './dataset_splits/ConGD/valid_rgb_list.txt'
model_prefix = '.'
weights_file = '%s/trained_models/tdres3d/%s_weights.{epoch:02d}-{val_loss:.2f}.h5'%(model_prefix,dataset_name)
  
train_data = data.load_con_video_list(training_datalist)
train_steps = len(train_data)/batch_size
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
print 'nb_epoch: %d - maxdepth: %d - batch_size: %d - weight_decay: %.6f' %(nb_epoch, depth, batch_size, weight_decay)

def lr_polynomial_decay(global_step):
  learning_rate = 0.0001
  end_learning_rate=0.000001
  decay_steps=train_steps*nb_epoch
  power = 0.9
  p = float(global_step)/float(decay_steps)
  lr = (learning_rate - end_learning_rate)*np.power(1-p, power)+end_learning_rate
  if global_step>0:
    curtime = '%s' % datetime.now()
    info = ' - lr: %.6f @ %s %d' %(lr, curtime.split('.')[0], global_step)
    print info,
  else:
    print 'learning_rate: %.6f - end_learning_rate: %.6f - decay_steps: %d' %(learning_rate, end_learning_rate, decay_steps)
  return lr
  
inputs = keras.layers.Input(shape=(None, 112, 112, 3))
feature = td_res3d(inputs, weight_decay)
prob = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), name='Bound_Prob')(feature)
outputs = keras.layers.Activation('sigmoid', name='Output')(prob)

model = keras.models.Model(inputs=inputs, outputs=outputs)
pretrained_model = '%s/trained_models/tdres3d/congr_tdres3d_rgb_weights.h5'%model_prefix
print 'Loading pretrained model from %s' % pretrained_model
model.load_weights(pretrained_model, by_name=False)
for i in range(len(model.trainable_weights)):
  print model.trainable_weights[i]

optimizer = keras.optimizers.SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='balanced_squared_hinge', metrics=['accuracy'])

lr_reducer = LearningRateScheduler(lr_polynomial_decay,train_steps) 
model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", 
                   save_best_only=False,save_weights_only=True,mode='auto')
callbacks = [lr_reducer, model_checkpoint]

model.fit_generator(conTrainImageBoundaryGenerator(training_datalist, 
                                        batch_size, depth, num_classes, RGB),
          steps_per_epoch=train_steps,
          epochs=nb_epoch,
          verbose=1,
          callbacks=callbacks,
          validation_data=conTestImageBoundaryGenerator(testing_datalist, 
                                        batch_size, depth, num_classes, RGB),
          validation_steps=test_steps,
          initial_epoch=init_epoch,
          )
