import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
import inputs as data
import threading

## Iteration
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batch_size]
    else:
      excerpt = slice(start_idx, start_idx + batch_size)
    yield inputs[excerpt], targets[excerpt]

## Threading
def threading_data(data=None, fn=None, **kwargs):
  # define function for threading
  def apply_fn(results, i, data, kwargs):
    results[i] = fn(data, **kwargs)

  ## start multi-threaded reading.
  results = [None] * len(data) ## preallocate result list
  threads = []
  for i in range(len(data)):
    t = threading.Thread(
                    name='threading_and_return',
                    target=apply_fn,
                    args=(results, i, data[i], kwargs)
                    )
    t.start()
    threads.append(t)

  ## <Milo> wait for all threads to complete
  for t in threads:
    t.join()

  return np.asarray(results)

## isoTrainImageGenerator
def isoTrainImageGenerator(filepath, batch_size, depth, num_classes, modality):
  X_train,y_train = data.load_iso_video_list(filepath)
  X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)
  y_train = np.asarray(y_train, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_tridx, y_train, 
                                            batch_size, shuffle=True):
      # Read data for each batch      
      image_path = []
      image_fcnt = []
      image_olen = []
      image_start = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_train[key_str]['videopath'])
        image_fcnt.append(X_train[key_str]['framecnt'])
        image_olen.append(depth)
        image_start.append(1)
        is_training.append(True) # Training
      image_info = zip(image_path,image_fcnt,image_olen,image_start,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_rgb_data)
      elif modality==1: #Depth
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_depth_data)
      elif modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_flow_data)     
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)
  
## isoTestImageGenerator
def isoTestImageGenerator(filepath, batch_size, depth, num_classes, modality):
  X_test,y_test = data.load_iso_video_list(filepath)
  X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
  y_test  = np.asarray(y_test, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_teidx, y_test, 
                                            batch_size, shuffle=False):
      # Read data for each batch      
      image_path = []
      image_fcnt = []
      image_olen = []
      image_start = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_test[key_str]['videopath'])
        image_fcnt.append(X_test[key_str]['framecnt'])
        image_olen.append(depth)
        image_start.append(1)
        is_training.append(False) # Testing
      image_info = zip(image_path,image_fcnt,image_olen,image_start,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_rgb_data)
      elif modality==1: #Depth
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_depth_data)
      elif modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_flow_data)     
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)

## conTrainImageGenerator
def conTrainImageGenerator(filepath, batch_size, depth, num_classes, modality):
  X_train = data.load_con_video_list(filepath)
  X_tridx = np.asarray(np.arange(0, len(X_train)), dtype=np.int32)
  while 1:
    for X_indices,_ in minibatches(X_tridx, X_tridx, 
                                   batch_size, shuffle=True):
      # Read data for each batch      
      image_path = []
      image_fcnt = []
      image_olen = []
      image_start = []
      is_training = []
      y_label_t = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        # Read data for each batch      
        idx = X_indices[data_a]
        video_path = X_train[idx].split(' ')[0]
        starti = int(X_train[idx].split(' ')[1].split(',')[0])
        endi = int(X_train[idx].split(' ')[1].split(',')[1].split(':')[0])
        label = int(X_train[idx].split(' ')[1].split(',')[1].split(':')[1])-1
        image_path.append(video_path)
        image_fcnt.append(endi-starti+1)
        image_olen.append(depth)
        image_start.append(starti)
        is_training.append(True) # Training
        y_label_t.append(label)
      image_info = zip(image_path,image_fcnt,image_olen,image_start,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_rgb_data)
      elif modality==1: #Depth
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_depth_data)
      elif modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_flow_data)     
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)
  
## conTestImageGenerator
def conTestImageGenerator(filepath, batch_size, depth, num_classes, modality):
  X_test = data.load_con_video_list(filepath)
  X_teidx = np.asarray(np.arange(0, len(X_test)), dtype=np.int32)
  while 1:
    for X_indices,_ in minibatches(X_teidx, X_teidx, 
                                   batch_size, shuffle=False):
      # Read data for each batch      
      image_path = []
      image_fcnt = []
      image_olen = []
      image_start = []
      is_training = []
      y_label_t = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        # Read data for each batch      
        idx = X_indices[data_a]
        video_path = X_test[idx].split(' ')[0]
        starti = int(X_test[idx].split(' ')[1].split(',')[0])
        endi = int(X_test[idx].split(' ')[1].split(',')[1].split(':')[0])
        label = int(X_test[idx].split(' ')[1].split(',')[1].split(':')[1])-1
        image_path.append(video_path)
        image_fcnt.append(endi-starti+1)
        image_olen.append(depth)
        image_start.append(starti)
        is_training.append(False) # Testing
        y_label_t.append(label)
      image_info = zip(image_path,image_fcnt,image_olen,image_start,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_rgb_data)
      elif modality==1: #Depth
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_depth_data)
      elif modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], 
                                data.prepare_iso_flow_data)     
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)

## conTrainImageBoundaryGenerator
def conTrainImageBoundaryGenerator(filepath, batch_size, depth, num_classes, modality):
  X_train = data.load_con_video_list(filepath)
  X_tridx = np.asarray(np.arange(0, len(X_train)), dtype=np.int32)
  while 1:
    for X_indices,_ in minibatches(X_tridx, X_tridx, 
                                            batch_size, shuffle=True):
      # Read data for each batch      
      video_label = []
      idx = X_indices[0]
      video_path = X_train[idx].split(' ')[0]
      segcnt = len(X_train[idx].split(' '))
      starti = endi = 0
      for i in range(1, segcnt):
        seginfo = X_train[idx].split(' ')[i]
        starti = int(seginfo.split(',')[0])
        if starti <= endi:
          starti = endi + 1
        endi = int(seginfo.split(',')[1].split(':')[0])
        label = int(seginfo.split(',')[1].split(':')[1])-1
        for j in range(starti, endi+1):
          video_label.append(label)
      if endi != len(video_label):
        print 'invalid: endi - %d, len(video_label) - %d'%(endi, len(video_label))
      video_fcnt = len(video_label)
      if len(video_label)<=depth:
        video_olen = len(video_label)
      else:
        video_olen = depth
      is_training = True # Training
      if modality==0: #RGB
        X_data_t,y_label = data.prepare_con_rgb_data(video_path, video_fcnt, video_olen, video_label, is_training)
      if modality==1: #Depth
        X_data_t,y_label = data.prepare_con_depth_data(video_path, video_fcnt, video_olen, video_label, is_training)
      if modality==2: #Flow
        X_data_t,y_label = data.prepare_con_flow_data(video_path, video_fcnt, video_olen, video_label, is_training)
      y_bound = np.zeros((len(y_label),), dtype=np.int32)
      for idx in range(2,len(y_label)-2):
        if y_label[idx-1]==y_label[idx] and y_label[idx+1]==y_label[idx+2] and y_label[idx]!=y_label[idx+1]:
          y_bound[idx-1]=1
          y_bound[idx]=1
          y_bound[idx+1]=1
          y_bound[idx+2]=1
      y_bound[0]=y_bound[1]=1
      y_bound[len(y_label)-1]=y_bound[len(y_label)-2]=1
      yield (np.reshape(X_data_t,(1,video_olen,112,112,3)), y_bound)
  
## conTestImageBoundaryGenerator
def conTestImageBoundaryGenerator(filepath, batch_size, depth, num_classes, modality):
  X_test = data.load_con_video_list(filepath)
  X_teidx = np.asarray(np.arange(0, len(X_test)), dtype=np.int32)
  while 1:
    for X_indices,_ in minibatches(X_teidx, X_teidx, 
                                            batch_size, shuffle=False):
      # Read data for each batch      
      video_label = []
      idx = X_indices[0]
      video_path = X_test[idx].split(' ')[0]
      segcnt = len(X_test[idx].split(' '))
      starti = endi = 0
      for i in range(1, segcnt):
        seginfo = X_test[idx].split(' ')[i]
        starti = int(seginfo.split(',')[0])
        if starti <= endi:
          starti = endi + 1
        endi = int(seginfo.split(',')[1].split(':')[0])
        label = int(seginfo.split(',')[1].split(':')[1])-1
        for j in range(starti, endi+1):
          video_label.append(label)
      if endi != len(video_label):
        print 'invalid: endi - %d, len(video_label) - %d'%(endi, len(video_label))
      video_fcnt = len(video_label)
      if len(video_label)<=depth:
        video_olen = len(video_label)
      else:
        video_olen = depth
      is_training = False # Testing
      if modality==0: #RGB
        X_data_t,y_label = data.prepare_con_rgb_data(video_path, video_fcnt, video_olen, video_label, is_training)
      if modality==1: #Depth
        X_data_t,y_label = data.prepare_con_depth_data(video_path, video_fcnt, video_olen, video_label, is_training)
      if modality==2: #Flow
        X_data_t,y_label = data.prepare_con_flow_data(video_path, video_fcnt, video_olen, video_label, is_training)
      y_bound = np.zeros((len(y_label),), dtype=np.int32)
      for idx in range(2,len(y_label)-2):
        if y_label[idx-1]==y_label[idx] and y_label[idx+1]==y_label[idx+2] and y_label[idx]!=y_label[idx+1]:
          y_bound[idx-1]=1
          y_bound[idx]=1
          y_bound[idx+1]=1
          y_bound[idx+2]=1
      y_bound[0]=y_bound[1]=1
      y_bound[len(y_label)-1]=y_bound[len(y_label)-2]=1
      yield (np.reshape(X_data_t,(1,video_olen,112,112,3)), y_bound)

## jesterTrainImageGenerator
def jesterTrainImageGenerator(filepath, batch_size, depth, num_classes, modality):
  X_train,y_train = data.load_iso_video_list(filepath)
  X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)
  y_train = np.asarray(y_train, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_tridx, y_train, 
                                            batch_size, shuffle=True):
      # Read data for each batch      
      image_path = []
      image_fcnt = []
      image_olen = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_train[key_str]['videopath'])
        image_fcnt.append(X_train[key_str]['framecnt'])
        image_olen.append(depth)
        is_training.append(True) # Training
      image_info = zip(image_path,image_fcnt,image_olen,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_rgb_data)
      if modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_flow_data)
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)
  
## jesterTestImageGenerator
def jesterTestImageGenerator(filepath, batch_size, depth, num_classes, modality):
  X_test,y_test = data.load_iso_video_list(filepath)
  X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
  y_test  = np.asarray(y_test, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_teidx, y_test, 
                                            batch_size, shuffle=False):
      # Read data for each batch      
      image_path = []
      image_fcnt = []
      image_olen = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_test[key_str]['videopath'])
        image_fcnt.append(X_test[key_str]['framecnt'])
        image_olen.append(depth)
        is_training.append(False) # Testing
      image_info = zip(image_path,image_fcnt,image_olen,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_rgb_data)
      if modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_flow_data)
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)

