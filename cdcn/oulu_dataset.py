#!/usr/bin/env python
# coding: utf-8

# In[13]:


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


# In[2]:


def filenameToPILImage(x): return Image.open(x)
# img_size = 224


def get_train_transforms(img_size=256, norm_mu=[0.485, 0.456, 0.406], norm_sig=[0.229, 0.224, 0.225]):
    return transforms.Compose([
        filenameToPILImage,
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.2, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mu, norm_sig)
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_gray_transforms():
    return transforms.Compose([
        filenameToPILImage,
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #         transforms.Normalize([0.449], [0.226])
    ])


def get_valid_transforms(img_size=256, norm_mu=[0.485, 0.456, 0.406], norm_sig=[0.229, 0.224, 0.225]):
    return transforms.Compose([
        filenameToPILImage,
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mu, norm_sig)
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class OULU(Dataset):
    def __init__(self, data_dir, mode, depth_dir='', vid_len=1, get_binary='label'):
        assert mode in [
            'train', 'valid', 'test'], 'only train mode, valid mode and test mode are avalible'
        assert get_binary in [
            'gray', 'label', 'depth'], 'The way to get binary mask should be gray, label or depth'

        # variable assignment
        self.data_dir = data_dir
        self.mode = mode
        self.vid_len = vid_len
        self.depth_dir = depth_dir
        self.get_binary = get_binary

        # get meta data and image transform method
        self.meta_data = sorted(os.listdir(data_dir))
        if self.mode == 'train':
            self.trans = get_train_transforms()
        else:
            self.trans = get_valid_transforms()

        self.gray_trans = get_gray_transforms()

        # the folder does not have depth map
        if mode == 'valid' and get_binary == 'depth':
            self.meta_data.remove('1_1_22_3')

        self.labels = [1 if self.meta_data[i].split(
            '_')[-1] == '1' else 0 for i in range(len(self.meta_data))]
        self.real_nums = sum(np.array(self.labels) == 1)
        self.fake_nums = sum(np.array(self.labels) == 0)
        self.weights = [self.fake_nums if self.labels[i] ==
                        1 else self.real_nums for i in range(len(self.labels))]

        # print out info
        print("OULU dataset is created!")
        print(f"Spec: \n mode:{self.mode} \n video length:{self.vid_len}")

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_size = 256
        video_path = os.path.join(self.data_dir, self.meta_data[idx])
        frame_names = sorted(os.listdir(video_path))

        assert len(
            frame_names) >= self.vid_len, "vid_len is too large for the dataset!"

        # set seed for this loading
        seed = randint(0, 2147483647)
        self.seed = seed

        # load frames
        data = torch.empty(3, self.vid_len, img_size, img_size)
        sample_list = sorted(random.sample(range(0, 11), self.vid_len))

        random.seed(seed)
        torch.manual_seed(seed)

        for i in range(self.vid_len):
            random.seed(seed)
            torch.manual_seed(seed)
            data[:, i, :, :] = self.trans(os.path.join(
                video_path, frame_names[sample_list[i]]))

        if self.vid_len == 1:
            data = data.squeeze(1)

        if self.mode == 'test':
            return data

        # set label
        label = torch.tensor(1) if self.meta_data[idx].split(
            '_')[-1] == '1' else torch.tensor(0)
        self.label = label

        # set binary mask
        binary_mask = self._get_binary_mask(idx, sample_list)
        return data, binary_mask,  label

    def _get_binary_mask(self, idx, sample_list):
        # generate from gray img => not worked
        # TODO: use np.zeors((32, 32)) for fake images
        if self.get_binary == 'gray':
            binary_mask = np.zeros((32, 32))
            gray_img = self.gray_trans(img_path).squeeze(0)
            for i in range(32):
                for j in range(32):
                    if gray_img[i, j] > 0.5:
                        binary_mask[i, j] = 1
                    else:
                        binary_mask[i, j] = 0

            return torch.FloatTensor(binary_mask)

        # generate binary mask from label
        elif self.get_binary == 'label':
            if self.label == 1:
                binary_mask = np.ones((32, 32))
            else:
                binary_mask = np.zeros((32, 32))
            return torch.FloatTensor(binary_mask)

        # generate binary mask from depth image
        elif self.get_binary == 'depth':
            if self.depth_dir == '':
                raise ValueError("depth dir should not be empty")

            try:
                # fake
                if self.label == 0:
                    binary_mask = np.zeros((32, 32))
                    binary_mask = torch.FloatTensor(binary_mask)
                    return binary_mask

                # modify to multiple frames for testing
                dep_video_path = os.path.join(
                    self.depth_dir, self.meta_data[idx])
                frame_names = sorted(os.listdir(dep_video_path))
                i = 0  # 0 for this case (video length = 1)
                dep_img_path = os.path.join(
                    dep_video_path, frame_names[sample_list[i]])

                if self.mode == 'train':
                    dep_trans = get_train_transforms(
                        img_size=32, norm_mu=[0.449], norm_sig=[0.226])
                else:
                    dep_trans = get_valid_transforms(
                        img_size=32, norm_mu=[0.449], norm_sig=[0.226])

                # fix seed for img and depth img
                random.seed(self.seed)
                torch.manual_seed(self.seed)

                dep_img = dep_trans(dep_img_path)
                binary_mask = torch.FloatTensor(dep_img).squeeze(0)

                return binary_mask

            except IndexError:
                # images failed to obtain depth image from PRnet -> use zeros by default
                binary_mask = np.zeros((32, 32))
                return torch.FloatTensor(binary_mask)


# In[3]:


# data_dir =  '../oulu_npu_cropped/train'
# depth_dir = '../oulu_npu_depth/train'
# dataset = OULU(data_dir, "train", depth_dir, get_binary='depth')
# data, m, label = dataset[131]
# print(data.shape)
# print(m.shape)
# print(len(dataset))


# In[14]:


# train_sampler = WeightedRandomSampler(dataset.weights, len(dataset), replacement=True)


# In[4]:


# import matplotlib.pyplot as plt

# plt.imshow(data.permute(1, 2, 0))
# plt.imshow(m)


# In[5]:


# data_dir =  '../oulu_npu_cropped/val'
# depth_dir = '../oulu_npu_depth/val'
# dataset = OULU(data_dir, "valid", depth_dir, vid_len=11, get_binary='depth')
# data, m, label = dataset[7]
# print(data.shape)
# print(m.shape)
# print(len(dataset))


# In[ ]:


# In[6]:


# d, b, l = dataset[1]
# print(b)


# In[7]:


# img_path = '../oulu_npu_cropped/train/1_1_01_2/015.png'
# trans = get_gray_transforms()
# gray_img = trans(img_path).squeeze(0)


# In[8]:


# import matplotlib.pyplot as plt

# binary_mask = np.zeros((32, 32))
# for i in range(32):
#     for j in range(32):
#         if gray_img[i, j]>0.5:
#             binary_mask[i,j]=1
#         else:
#             binary_mask[i,j]=0

# print(np.mean(binary_mask))

# plt.imshow(binary_mask)


# In[ ]:


# In[ ]:
