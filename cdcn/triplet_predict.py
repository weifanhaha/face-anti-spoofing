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
# from cdcn import Conv2d_cd, CDCN, CDCNpp
from cdcn_paper import Conv2d_cd, CDCN, CDCNpp
from oulu_dataset import OULU
from siw_dataset import SIW


# In[2]:


# load dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

val_image_dir = '../oulu_npu_cropped/val'
test_image_dir = '../oulu_npu_cropped/test'

val_dataset = OULU(val_image_dir, "valid", vid_len=11, get_binary='label')
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=4)

test_dataset = OULU(test_image_dir, "test", vid_len=11, get_binary='label')
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=4)

output_csv = './outputs/cdcnpp_paper_depth_triplet.csv'
output_siw_csv = './outputs/siw_cdcnpp_paper_depth_triplet.csv'


# In[3]:


finetune_model_path = './models/cdcnpp_paper_depth_triplet.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finetuned_dict = torch.load(finetune_model_path, map_location=device)


# In[4]:


# load pretrained cvcnpp model
theta = 0.7
model = CDCNpp(basic_conv=Conv2d_cd, theta=theta).to(device)
model_dict = model.state_dict()

for name, param in finetuned_dict.items():
    if name not in model_dict:
        print("skip name ", name)
        continue

    param = param.data
    model_dict[name].copy_(param)

model.load_state_dict(model_dict)


# In[5]:


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
                map_x, embedding, x_Block1, x_Block2, x_Block3, x_input = model(
                    data[:, :, frame, :, :])
                map_score = torch.mean(map_x)
                sum_map_score += map_score

        map_score = sum_map_score / data.shape[2]
        map_score = map_score.item() / 2
#         map_score = 1.0 if map_score > 1 else map_score.item()
        map_score_list.append(map_score)

        pred = 1 if map_score > 0.5 else 0
        acc += (pred == label.item())

    return acc / len(val_dataset), map_score_list


# In[6]:


acc, map_score_list = eval_one_epoch()
print("Accuracy in oulu val set: ", acc)


# In[7]:


labels = []
for fname in sorted(os.listdir(val_image_dir)):
    video_type = int(fname.split('_')[-1])
    label = 1 if video_type == 1 else 0
    labels.append(label)


# In[8]:


auc = metrics.roc_auc_score(labels, map_score_list)
print("AUC in oulu val set: ", auc)


# In[9]:


# generate test data
model.eval()
map_score_list = []

for i, data in enumerate(tqdm(test_dataloader)):
    # get the inputs
    with torch.no_grad():
        data = data.cuda()
        sum_map_score = 0.0

        for frame in range(data.shape[2]):  # number of frames
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input = model(
                data[:, :, frame, :, :])
            map_score = torch.mean(map_x)
            sum_map_score += map_score

    map_score = sum_map_score / data.shape[2]
    map_score = 1.0 if map_score > 1 else map_score.item()
    map_score_list.append(map_score)


# In[10]:


# write to csv
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['video_id', 'label'])

    for i, score in enumerate(map_score_list):
        vid = '{0:04d}'.format(i)
        writer.writerow([vid, score])


# In[11]:


# predict SiW dataset
data_dir = '../siw_test'
siw_dataset = SIW(data_dir, "test", vid_len=10)
siw_dataloader = DataLoader(
    siw_dataset, batch_size=1, shuffle=False, num_workers=4)


# In[12]:


# generate test data
model.eval()
map_score_list = []

for i, data in enumerate(tqdm(siw_dataloader)):
    # get the inputs
    with torch.no_grad():
        data = data.cuda()
        sum_map_score = 0.0

        for frame in range(data.shape[2]):  # number of frames
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input = model(
                data[:, :, frame, :, :])
            map_score = torch.mean(map_x)
            sum_map_score += map_score

    map_score = sum_map_score / data.shape[2]
    map_score = 1.0 if map_score > 1 else map_score.item()
    map_score_list.append(map_score)


# In[13]:


# write to csv
with open(output_siw_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['video_id', 'label'])

    for i, score in enumerate(map_score_list):
        vid = '{0:04d}'.format(i)
        writer.writerow([vid, score])


# In[ ]:
