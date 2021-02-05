#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import copy
from tqdm import tqdm
from sklearn import metrics

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from utils import AvgrageMeter, accuracy, performances, FeatureMap2Heatmap
from cdcn_paper import Conv2d_cd, CDCNpp, SpatialAttention
from oulu_dataset import OULU
from hard_triplet_loss import HardTripletLoss


# In[2]:


class CDCNppTriplet(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCNppTriplet, self).__init__()

        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            basic_conv(128, int(128*1.6), kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128*1.6)),
            nn.ReLU(),
            basic_conv(int(128*1.6), 128, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )

        self.Block2 = nn.Sequential(
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, int(128*1.4), kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128*1.4)),
            nn.ReLU(),
            basic_conv(int(128*1.4), 128, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Original

        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 1, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.ReLU(),
        )

        self.sa1 = SpatialAttention(kernel=7)
        self.sa2 = SpatialAttention(kernel=5)
        self.sa3 = SpatialAttention(kernel=3)
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

    def forward(self, x):	    	# x [3, 256, 256]

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)

        x_Block2 = self.Block2(x_Block1)
        attention2 = self.sa2(x_Block2)
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)

        x_Block3 = self.Block3(x_Block2)
        attention3 = self.sa3(x_Block3)
        x_Block3_SA = attention3 * x_Block3
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)

        x_concat = torch.cat(
            (x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)

        # pdb.set_trace()

        map_x = self.lastconv1(x_concat)
        map_x = map_x.squeeze(1)

        triplet_out = map_x.view(map_x.shape[0], -1)

        feature_norm = torch.norm(triplet_out, p=2, dim=1, keepdim=True).clamp(
            min=1e-12) ** 0.5 * (2) ** 0.5
        triplet_out = torch.div(triplet_out, feature_norm)

        return map_x, x_concat, attention1, attention2, attention3, x_input, triplet_out


# In[3]:


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0],
                                             [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]
         ], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1,
                                                         0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(
        kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)

    input = input.unsqueeze(dim=1).expand(
        input.shape[0], 8, input.shape[1], input.shape[2])

    contrast_depth = F.conv2d(
        input, weight=kernel_filter, groups=8)  # depthwise conv
#     contrast_depth = F.conv2d(input.float(), weight=kernel_filter, groups=8)  # depthwise conv

    return contrast_depth


# In[4]:


# Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
class Contrast_depth_loss(nn.Module):
    def __init__(self):
        super(Contrast_depth_loss, self).__init__()
        return

    def forward(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)

        criterion_MSE = nn.MSELoss().cuda()

        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)

        return loss


# In[5]:


# parse arguments
# paper code
parser = argparse.ArgumentParser(
    description="save quality using landmarkpose model")
parser.add_argument('--lr', type=float, default=0.00008,
                    help='initial learning rate')
parser.add_argument('--batchsize', type=int, default=12,
                    help='initial batchsize')
parser.add_argument('--step_size', type=int, default=40,
                    help='how many epochs lr decays once')  # 500
parser.add_argument('--gamma', type=float, default=0.5,
                    help='gamma of optim.lr_scheduler.StepLR, decay of lr')
parser.add_argument('--echo_batches', type=int, default=20,
                    help='how many batches display once')  # 50
parser.add_argument('--epochs', type=int, default=60,
                    help='total training epochs')  # 1400
parser.add_argument('--log', type=str, default="CDCNpp_P1",
                    help='log and save model name')
parser.add_argument('--finetune', action='store_true',
                    default=False, help='whether finetune other models')
parser.add_argument('--theta', type=float, default=0.7,
                    help='hyper-parameters in CDCNpp')

args = parser.parse_args("")


# In[6]:


# load dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_image_dir = '../oulu_npu_cropped/train'
val_image_dir = '../oulu_npu_cropped/val'

train_depth_dir = '../oulu_npu_depth/train'
val_depth_dir = '../oulu_npu_depth/val'

get_binary = 'depth'

train_dataset = OULU(train_image_dir, "train",
                     train_depth_dir, get_binary=get_binary)
train_sampler = WeightedRandomSampler(
    train_dataset.weights, len(train_dataset), replacement=True)
# train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, num_workers=4, shuffle=True)
train_dataloader = DataLoader(
    train_dataset, batch_size=args.batchsize, num_workers=4, sampler=train_sampler)


val_dataset = OULU(val_image_dir, "valid",
                   val_depth_dir, get_binary=get_binary)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=4)


# In[7]:


_, _, data = next(iter(train_dataloader))
data


# In[8]:


labels = []
for fname in sorted(os.listdir(val_image_dir)):
    # for depth map
    if get_binary == 'depth' and fname == '1_1_22_3':
        continue
    video_type = int(fname.split('_')[-1])
    l = 1 if video_type == 1 else 0
    labels.append(l)


# In[9]:


pretrained_model_path = './models/cdcnpp_paper_depth_best.pth'
finetune_model_path = './models/cdcnpp_paper_depth_triplet.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_dict = torch.load(pretrained_model_path, map_location=device)


# In[10]:


# load pretrained cvcnpp model
model = CDCNppTriplet(basic_conv=Conv2d_cd, theta=args.theta).to(device)
model_dict = model.state_dict()

for name, param in pretrained_dict.items():
    if name not in model_dict:
        print("skip name ", name)
        continue

    param = param.data
    model_dict[name].copy_(param)

model.load_state_dict(model_dict)


# In[11]:


lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=args.step_size, gamma=args.gamma)

# init loss
criterion_absolute_loss = nn.MSELoss().cuda()
criterion_contrastive_loss = Contrast_depth_loss().cuda()
criterion_triplet_loss = HardTripletLoss(margin=0.1, hardest=False).cuda()


# In[12]:


def train_one_epoch():
    model.train()
    loss_absolute = AvgrageMeter()
    loss_contra = AvgrageMeter()
    loss_triplet = AvgrageMeter()

    trange = tqdm(train_dataloader)

    for i, batch in enumerate(trange):
        # get the inputs
        data, binary_mask, label = batch
        data, binary_mask, label = data.cuda(), binary_mask.cuda(), label.cuda()

        optimizer.zero_grad()

        # forward + backward + optimize
        map_x, _, _, _, _, _, triplet_out = model(data)

        absolute_loss = criterion_absolute_loss(map_x, binary_mask)
        contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
        triplet_loss = criterion_triplet_loss(triplet_out, label)

        loss = absolute_loss + contrastive_loss + triplet_loss
        loss.backward()
        optimizer.step()

        n = data.size(0)
        loss_absolute.update(absolute_loss.data, n)
        loss_contra.update(contrastive_loss.data, n)
        loss_triplet.update(triplet_loss.data, n)

        postfix_dict = {
            "loss_absolute": absolute_loss.item(),
            "loss_contra": contrastive_loss.item(),
            "loss_triplet": triplet_loss.item(),
            "loss": loss.item()
        }
        trange.set_postfix(**postfix_dict)

    print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f, Triplet_loss= %.8f\n' % (
        epoch + 1, loss_absolute.avg, loss_contra.avg, loss_triplet.avg)
    )


# In[13]:


# eval one epoch
def eval_one_epoch():
    model.eval()
    acc = 0.0
    map_score_list = []

    loss_absolute = AvgrageMeter()
    loss_contra = AvgrageMeter()
    loss_triplet = AvgrageMeter()

    for i, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs
        with torch.no_grad():
            data, binary_mask, label = batch
            data, binary_mask, label = data.cuda(), binary_mask.cuda(), label.cuda()

            optimizer.zero_grad()
            map_score = 0.0

            map_x, _, _, _, _, _, triplet_out = model(data)

            absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            triplet_loss = criterion_triplet_loss(triplet_out, label)

            loss = absolute_loss + contrastive_loss + triplet_loss

            n = data.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)
            loss_triplet.update(triplet_loss.data, n)

            map_score = torch.mean(map_x)

        map_score = 1.0 if map_score > 1 else map_score.item()
        map_score_list.append(map_score)

        # need another way to evaluate
        pred = 1 if map_score > 0.5 else 0
        acc += (pred == label.item())

    loss_avg = loss_absolute.avg + loss_contra.avg
    aou = metrics.roc_auc_score(labels, map_score_list)

    print('epoch:%d, Eval:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f, Triplet_loss: %.8f, Total Loss: %.4f, AOU: %.4f\n' % (
        epoch + 1, loss_absolute.avg, loss_contra.avg, loss_triplet.avg, loss_avg, aou)
    )

    return acc / len(val_dataset),  loss_avg, aou, map_score_list


# In[ ]:


best_model = None
best_aou = 0.0
best_loss = 999

for epoch in range(args.epochs):
    print("Epoch {} / {}".format(epoch+1, args.epochs))
    model.train()
    scheduler.step()
    train_one_epoch()

    # eval every 2 epochs
    if (epoch + 1) % 2 == 0:
        model.eval()
        acc, loss_avg, aou, map_score_list = eval_one_epoch()
        print(map_score_list[:10])

#         if aou >= best_aou:
#             best_aou = aou
#             best_model = copy.deepcopy(model.state_dict())

        # use loss to select model
        if loss_avg < best_loss:
            best_loss = loss_avg
            best_model = copy.deepcopy(model.state_dict())

    if (epoch + 1) % 10 == 0:
        torch.save(best_model, finetune_model_path)
        print("model saved")
