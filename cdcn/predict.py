#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import torch
import numpy as np
from sklearn import metrics
import csv

from utils import AvgrageMeter, accuracy, performances, FeatureMap2Heatmap
from cdcn import Conv2d_cd, CDCN, CDCNpp
from oulu_dataset import OULU


# In[2]:


# load dataset
os.environ["CUDA_VISIBLE_DEVICES"]="1"

val_image_dir = '../oulu_npu_cropped/val'
test_image_dir = '../oulu_npu_cropped/test'

val_dataset = OULU(val_image_dir, "valid", vid_len=11, get_binary='label')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

test_dataset = OULU(test_image_dir, "test", vid_len=11, get_binary='label')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


# model_path = './models/cdcnpp.pth'
# output_csv = './outputs/cdcnpp.csv'

model_path = './models/cdcnpp_label.pth'
output_csv = './outputs/cdcnpp_label_frames.csv'


# In[6]:


batch[0][:,:,0,:,:].shape


# In[10]:


# init model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
theta = 0.7

model = CDCNpp(basic_conv=Conv2d_cd, theta=theta)
model = model.cuda()

model.load_state_dict(torch.load(
        model_path, map_location=device))


# In[18]:


# eval one epoch
def eval_one_epoch():
    model.eval()
    acc = 0.0
    best_acc = 0.0
    best_model = None
    map_score_list = []

    for i, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs
        with torch.no_grad():
            data, binary_mask, label = batch
            data, binary_mask, label = data.cuda(), binary_mask.cuda(), label.cuda()
            sum_map_score = 0.0
            
            for frame in range(data.shape[2]):  # number of frames
                map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(data[:,:,frame,:,:])
                map_score = torch.mean(map_x)
                sum_map_score += map_score

        map_score = sum_map_score / data.shape[2]
        map_score = 1.0 if map_score > 1 else map_score.item()
        map_score_list.append(map_score)

        pred = 1 if map_score > 0.5 else 0
        acc += (pred == label.item())
        

    return acc / len(val_dataset), map_score_list


# In[19]:


acc, map_score_list = eval_one_epoch()


# In[20]:


acc


# In[21]:


labels = []
for fname in sorted(os.listdir(val_image_dir)):
    video_type = int(fname.split('_')[-1])
    label = 1 if video_type == 1 else 0
    labels.append(label)
# video_path = os.path.join(self.data_dir, self.meta_data[idx])
# frame_names = sorted(os.listdir(video_path))


# In[22]:


metrics.roc_auc_score(labels, map_score_list)


# In[23]:


# generate test data
model.eval()
map_score_list = []

for i, data in enumerate(tqdm(test_dataloader)):
    # get the inputs
    with torch.no_grad():
        data = data.cuda()
        sum_map_score = 0.0

        for frame in range(data.shape[2]):  # number of frames
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(data[:,:,frame,:,:])
            map_score = torch.mean(map_x)
            sum_map_score += map_score

    map_score = sum_map_score / data.shape[2]
    map_score = 1.0 if map_score > 1 else map_score.item()
    map_score_list.append(map_score)


# In[24]:


# write to csv
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['video_id', 'label'])
    
    for i, score in enumerate(map_score_list):
        vid = '{0:04d}'.format(i)
        writer.writerow([vid, score])


# In[ ]:




