import io
import os
import sys
import math
import random
import numpy as np
from scipy.misc import imread, imresize

def load_con_video_list(path):
  assert os.path.exists(path)
  f = open(path, 'r')
  f_lines = f.readlines()
  f.close()
  video_data = [] 
  for idx, line in enumerate(f_lines):
    video_data.append(line)
  return video_data

TESTING = True
VALIDATING = False

if VALIDATING==True:
  rgb_prediction = np.load('features/rgb_valid_boundscore_tdres3d.npy')
  depth_prediction = np.load('features/depth_valid_boundscore_tdres3d.npy')
  testing_datalist = './dataset_splits/ConGD/valid_rgb_list.txt'

if TESTING==True:
  rgb_prediction = np.load('features/rgb_test_boundscore_tdres3d.npy')
  depth_prediction = np.load('features/depth_test_boundscore_tdres3d.npy')
  testing_datalist = './dataset_splits/ConGD/test_rgb_list.txt'

#fusion_prediction = np.sqrt(rgb_prediction*depth_prediction)
fusion_prediction = (rgb_prediction + depth_prediction)/2
average_prediction = fusion_prediction 
for i in range(1, len(fusion_prediction)-1):
  average_prediction[i] = (fusion_prediction[i-1]+fusion_prediction[i]*3+fusion_prediction[i+1])/5
boundary = average_prediction[:,]>=0.5

depth = 128
test_data = load_con_video_list(testing_datalist)
test_steps = len(test_data)
total_offset = 0
for idx in range(test_steps):
  video_path = test_data[idx].split(' ')[0]
  segcnt = len(test_data[idx].split(' '))
  starti = endi = 0
  video_label = []
  for i in range(1, segcnt):
    seginfo = test_data[idx].split(' ')[i]
    starti = int(seginfo.split(',')[0])
    if starti <= endi:
      starti = endi + 1
    endi = int(seginfo.split(',')[1].split(':')[0])
    label = int(seginfo.split(',')[1].split(':')[1])-1
    for j in range(starti, endi+1):
      video_label.append(label)
  video_len = len(video_label)
  if video_len < depth:
    rand_frames = np.arange(1, video_len+1)
  else:
    div = float(video_len)/float(depth)
    rand_frames = np.zeros(depth)
    rand_frames[::] = div*np.arange(0, depth)
    rand_frames[0] = max(rand_frames[0], 0)
    rand_frames[depth-1] = min(rand_frames[depth-1], video_len-1)
    rand_frames = np.floor(rand_frames)+1
  predict_len = len(rand_frames)
  predict_cnt = 1
  bound_len = 0
  start_idx = []
  end_idx = []
  start_idx.append(1)
  for pos in range(total_offset+2, total_offset+predict_len-2):
    if boundary[pos]==True:
      bound_len = bound_len+1
    else:
      if bound_len > 1:
        bound_idx1 = rand_frames[pos-total_offset-bound_len/2-1]
        bound_idx2 = rand_frames[pos-total_offset-bound_len/2]
        if bound_idx1+1!=bound_idx2:
          bound_idx1=(bound_idx1+bound_idx2)/2
          bound_idx2=bound_idx1+1
        end_idx.append(bound_idx1)
        start_idx.append(bound_idx2)
        predict_cnt = predict_cnt + 1
      bound_len = 0
  end_idx.append(video_len)
  total_offset = total_offset+predict_len
  for i in range(len(start_idx)):
    if i==len(start_idx)-1:
      print '%s %d,%d:0' % (video_path, start_idx[i], end_idx[i]) #RGB&Depth
      #print '%s %d,%d:0' % (video_path, start_idx[i], end_idx[i]-1) #Flow
    else:
      print '%s %d,%d:0' % (video_path, start_idx[i], end_idx[i])
