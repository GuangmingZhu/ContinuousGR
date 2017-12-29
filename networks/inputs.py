import io
import os
import sys
import math
import random
import numpy as np
from scipy.misc import imread, imresize


def load_iso_video_list(path):
  assert os.path.exists(path)
  f = open(path, 'r')
  f_lines = f.readlines()
  f.close()
  video_data = {} 
  video_label = []
  for idx, line in enumerate(f_lines):
    video_key = '%06d' % idx
    video_data[video_key] = {} 
    videopath  = line.split(' ')[0]
    framecnt   = int(line.split(' ')[1])
    videolabel = int(line.split(' ')[2])
    video_data[video_key]['videopath'] = videopath
    video_data[video_key]['framecnt'] = framecnt
    video_label.append(videolabel)
  return video_data,video_label

def prepare_ucf_rgb_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = 16
  start_frame_idx = image_info[1]
  output_frame_cnt = image_info[2]
  is_training = image_info[3]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt) 
  scale = math.floor(div)
  if is_training:
    if scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 0)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt-1)
  rand_frames = np.floor(rand_frames)+start_frame_idx

  average_values = [128,128,128]
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in xrange(0, output_frame_cnt):
    image_file = '%s%04d.jpg' %(video_path, rand_frames[idx])
    assert os.path.exists(image_file)
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    szimage = imresize(image, (128,171)) 
    square_sz = 112
    if is_training:
      crop_h = int((128 - square_sz)*crop_random)
      crop_w = int((171 - square_sz)*crop_random)
    else:
      crop_h = int((128 - square_sz)/2)
      crop_w = int((171 - square_sz)/2)
    image_crop = szimage[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = image_crop - average_values
  return processed_images

def prepare_skig_rgb_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]
  output_frame_cnt = image_info[2]
  is_training = image_info[3]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(1, video_frame_cnt+1)
      rand_frames[video_frame_cnt::] = video_frame_cnt
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(1, video_frame_cnt+1)
      rand_frames[video_frame_cnt::] = video_frame_cnt
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 1)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt)
  rand_frames = np.floor(rand_frames)

  average_values = [132,112,96]
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in xrange(0,output_frame_cnt):
    image_file = '%s/%04d.jpg' %(video_path, rand_frames[idx])
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def prepare_skig_depth_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]
  output_frame_cnt = image_info[2]
  is_training = image_info[3]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(1, video_frame_cnt+1)
      rand_frames[video_frame_cnt::] = video_frame_cnt
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(1, video_frame_cnt+1)
      rand_frames[video_frame_cnt::] = video_frame_cnt
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 1)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt)
  rand_frames = np.floor(rand_frames)

  average_values = [237,237,237]
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in xrange(0,output_frame_cnt):
    image_file = '%s/%04d.jpg' %(video_path, rand_frames[idx])
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def prepare_skig_flow_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]-1
  output_frame_cnt = image_info[2]
  is_training = image_info[3]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(1, video_frame_cnt+1)
      rand_frames[video_frame_cnt::] = video_frame_cnt
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(1, video_frame_cnt+1)
      rand_frames[video_frame_cnt::] = video_frame_cnt
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 1)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt)
  rand_frames = np.floor(rand_frames)

  average_values = [128,128,128] 
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in xrange(0,output_frame_cnt):
    image_file = '%s/%04d.jpg' %(video_path, rand_frames[idx])
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def prepare_iso_rgb_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]
  output_frame_cnt = image_info[2]
  start_frame_idx = image_info[3]
  is_training = image_info[4]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 0)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt-1)
  rand_frames = np.floor(rand_frames)+start_frame_idx

  average_values = [112,112,112]
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in xrange(0, output_frame_cnt):
    image_file = '%s/%06d.jpg' %(video_path, rand_frames[idx])
    assert os.path.exists(image_file)
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def prepare_iso_depth_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]
  output_frame_cnt = image_info[2]
  start_frame_idx = image_info[3]
  is_training = image_info[4]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 0)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt-1)
  rand_frames = np.floor(rand_frames)+start_frame_idx

  average_values = [127,127,127] 
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in xrange(0, output_frame_cnt):
    image_file = '%s/%06d.jpg' %(video_path, rand_frames[idx])
    assert os.path.exists(image_file)
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def prepare_iso_flow_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]
  output_frame_cnt = image_info[2]
  start_frame_idx = image_info[3]
  is_training = image_info[4]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 0)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt-1)
  rand_frames = np.floor(rand_frames)+start_frame_idx

  average_values = [128,128,128] 
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in xrange(0, output_frame_cnt):
    image_file = '%s/%06d.jpg' %(video_path, rand_frames[idx])
    assert os.path.exists(image_file)
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def prepare_jester_rgb_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]
  output_frame_cnt = image_info[2]
  is_training = image_info[3]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 0)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt-1)
  rand_frames = np.floor(rand_frames)+1

  average_values = [114,109,104]
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in xrange(0, output_frame_cnt):
    image_file = '%s/%05d.jpg' %(video_path, rand_frames[idx])
    assert os.path.exists(image_file)
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def prepare_jester_flow_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]-1
  output_frame_cnt = image_info[2]
  is_training = image_info[3]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 0)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt-1)
  rand_frames = np.floor(rand_frames)+1

  average_values = [128,128,128] 
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in xrange(0, output_frame_cnt):
    image_file = '%s/%06d.jpg' %(video_path, rand_frames[idx])
    assert os.path.exists(image_file)
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def load_con_video_list(path):
  assert os.path.exists(path)
  f = open(path, 'r')
  f_lines = f.readlines()
  f.close()
  video_data = [] 
  for idx, line in enumerate(f_lines):
    video_data.append(line)
  return video_data

def prepare_con_rgb_data(video_path, video_fcnt, video_olen, video_label, is_training):
  video_frame_cnt = video_fcnt
  output_frame_cnt = video_olen
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 1:
      rand_frames[::] = np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 0)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt-1)
  rand_frames = np.floor(rand_frames)+1

  average_values = [112,112,112]
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  image_label = np.zeros((output_frame_cnt,), dtype=np.int32)
  crop_random = random.random()
  for idx in xrange(0, output_frame_cnt):
    image_file = '%s/%06d.jpg' %(video_path, rand_frames[idx])
    image_label[idx] = video_label[int(rand_frames[idx])-1]
    try:
      assert os.path.exists(image_file)
    except:
      print '%s does not exist' % image_file
      assert os.path.exists(image_file)
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images,image_label

def prepare_con_depth_data(video_path, video_fcnt, video_olen, video_label, is_training):
  video_frame_cnt = video_fcnt
  output_frame_cnt = video_olen
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 1:
      rand_frames[::] = np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 0)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt-1)
  rand_frames = np.floor(rand_frames)+1

  average_values = [127,127,127] 
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  image_label = np.zeros((output_frame_cnt,), dtype=np.int32)
  crop_random = random.random()
  for idx in xrange(0, output_frame_cnt):
    image_file = '%s/%06d.jpg' %(video_path, rand_frames[idx])
    image_label[idx] = video_label[int(rand_frames[idx])-1]
    try:
      assert os.path.exists(image_file)
    except:
      print '%s does not exist' % image_file
      assert os.path.exists(image_file)
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images,image_label

