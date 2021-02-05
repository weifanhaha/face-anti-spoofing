#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.transforms as transforms
from torch.utils.data import Dataset
from random import randint
from PIL import Image
import torch
import os
import math
import random
import numpy as np
from random import randint


# In[8]:



filenameToPILImage = lambda x: Image.open(x)
# img_size = 224


def get_gray_transforms():
    return transforms.Compose([
        filenameToPILImage,
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

def get_valid_transforms(img_size=256, norm_mu=[0.485, 0.456, 0.406], norm_sig=[0.229, 0.224, 0.225]):
    return transforms.Compose([
        filenameToPILImage,
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mu, norm_sig)
    ])

class SIW(Dataset):
    def __init__(self, data_dir, mode, depth_dir='', vid_len=1, get_binary='label'):
        assert mode in ['test'], 'only test mode is avalible'
        assert get_binary in ['gray', 'label', 'depth'], 'The way to get binary mask should be gray, label or depth'
        
        # variable assignment
        self.data_dir = data_dir
        self.mode = mode
        self.vid_len = vid_len
        self.depth_dir = depth_dir
        self.get_binary = get_binary
        
        # get meta data and image transform method
        self.meta_data = sorted(os.listdir(data_dir))
        self.trans = get_valid_transforms()
        
        self.gray_trans = get_gray_transforms()
        
        # print out info
        print("SIW dataset is created!")
        print(f"Spec: \n mode:{self.mode} \n video length:{self.vid_len}")
    
    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        img_size = 256
        video_path = os.path.join(self.data_dir, self.meta_data[idx])
        frame_names = sorted(os.listdir(video_path))
        
        assert len(frame_names) >= self.vid_len, "vid_len is too large for the dataset!"
        
        # set seed for this loading
        seed = randint(0, 2147483647)
        self.seed = seed
        
        # load frames
        data = torch.empty(3, self.vid_len, img_size, img_size)
        sample_list = sorted(random.sample(range(0, 10), self.vid_len))

        random.seed(seed) 
        torch.manual_seed(seed)

        for i in range(self.vid_len):
            random.seed(seed) 
            torch.manual_seed(seed)
            data[:,i,:,:] = self.trans(os.path.join(video_path, frame_names[sample_list[i]]))

        if self.vid_len == 1:
            data = data.squeeze(1)
        
        return data


# In[9]:


# data_dir =  '../siw_test'
# dataset = SIW(data_dir, "test", vid_len=10)
# data = dataset[7]
# print(data.shape)


# In[ ]:




