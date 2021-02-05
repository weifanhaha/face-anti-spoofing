#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch import nn
from torch.nn import Parameter
import torch.optim as optim

import pdb
import numpy as np
from sklearn import metrics

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import copy
from tqdm import tqdm

from oulu_dataset import OULU
from cdcn_paper import Conv2d_cd, SpatialAttention


# In[2]:


class CDCNppClassifier(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7 ):   
        super(CDCNppClassifier, self).__init__()
        
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
            
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            
            basic_conv(128, int(128*1.6), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.6)),
            nn.ReLU(),  
            basic_conv(int(128*1.6), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),  
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            basic_conv(128, int(128*1.4), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.4)),
            nn.ReLU(),  
            basic_conv(int(128*1.4), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),  
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Original
        
        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        
      
        self.sa1 = SpatialAttention(kernel = 7)
        self.sa2 = SpatialAttention(kernel = 5)
        self.sa3 = SpatialAttention(kernel = 3)
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')
        
        self.fc = nn.Linear(32*32, 2, bias=True)

 
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
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    
        
        #pdb.set_trace()
        
        map_x = self.lastconv1(x_concat)
        
        map_x = map_x.squeeze(1)
        
        features = map_x.flatten(start_dim=1)
        score = self.fc(features)
        
        return score
        
#         return map_x, x_concat, attention1, attention2, attention3, x_input


# In[3]:


# parse arguments
# paper code
import argparse
parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--batchsize', type=int, default=3, help='initial batchsize')
parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')  # 500
parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
parser.add_argument('--epochs', type=int, default=60, help='total training epochs')
parser.add_argument('--log', type=str, default="CDCNpp_P1", help='log and save model name')
parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')

args = parser.parse_args("")


# In[4]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[5]:


pretrained_model_path = './models/cdcnpp_paper_depth.pth' 
finetune_model_path = './models/cdcnpp_paper_depth_finetune.pth' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_dict = torch.load(pretrained_model_path, map_location=device)


# In[6]:


# load pretrained cvcnpp model
model = CDCNppClassifier(basic_conv=Conv2d_cd, theta=args.theta).to(device)
model_dict = model.state_dict()

for name, param in pretrained_dict.items():
    if name not in model_dict:
        print("skip name ", name)
        continue

    param = param.data
    model_dict[name].copy_(param)

model.load_state_dict(model_dict)


# In[7]:


# load dataset
train_image_dir = '../oulu_npu_cropped/train'
val_image_dir = '../oulu_npu_cropped/val'

train_depth_dir = '../oulu_npu_depth/train'
val_depth_dir = '../oulu_npu_depth/val'

train_dataset = OULU(train_image_dir, "train", train_depth_dir, get_binary='depth')
train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)

val_dataset = OULU(val_image_dir, "valid", val_depth_dir, get_binary='depth')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)


# In[8]:


num_epochs = 60
lr = 0.0001
step_size = 20
gamma = 0.5
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# init loss
criterion = nn.CrossEntropyLoss().cuda()


# In[9]:


# # check runable
# data, binary_mask, label = next(iter(train_dataloader))
# model = CDCNppClassifier(basic_conv=Conv2d_cd, theta=args.theta)
# output = model(data)
# print(output)

# loss = criterion(output, label)
# print(loss)


# In[10]:


# torch.max(output, 1)


# In[44]:


def train_one_epoch():
    model.train()
    trange = tqdm(train_dataloader)
    
    loss_sum = 0.0
    acc_sum = 0.0
    
    softmax = nn.Softmax(dim=1)

    for i, batch in enumerate(trange):
        # get the inputs
        data, _, label = batch
        data, label = data.cuda(), label.cuda()

        optimizer.zero_grad()
        
        # forward + backward + optimize
        score = model(data)
        score = softmax(score)
        loss = criterion(score, label)
        
        _, pred = torch.max(score, 1)

        # need another way to evaluate
        acc = sum(pred == label).item()
        acc_sum += acc

        loss.backward()
        optimizer.step()

        postfix_dict = {
            "loss": loss.item(),
            "acc": acc_sum / (data.shape[0] * (i+1))
        }
        trange.set_postfix(**postfix_dict)
        loss_sum += loss.item()

    print('epoch:%d, Train:  Loss= %.4f, Train Acc=%4f\n' % (epoch + 1, loss_sum / len(trange), acc_sum / len(train_dataset)))


# In[45]:


# eval one epoch
def eval_one_epoch():
    model.eval()
    acc = 0.0
    loss_sum = 0.0
    best_acc = 0.0
    best_model = None
    score_list = []
    softmax = nn.Softmax(dim=1)

    for i, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs
        with torch.no_grad():
            data, _, label = batch
            data, label = data.to(device), label.to(device)
            
            optimizer.zero_grad()
            score = model(data)
            score = softmax(score)
            
            loss = criterion(score, label)
            loss_sum += loss.item()
            
            _, pred = torch.max(score, 1)

        score_list.append(score[0][1].item())
    
        # need another way to evaluate
        acc += sum(pred == label).item()


    
    print('epoch:%d, Eval:  Loss= %.4f, Eval Acc=%4f\n' % (epoch + 1, loss_sum / len(val_dataloader), acc / len(val_dataset)))
    return acc / len(val_dataset), score_list


# In[46]:


labels = []
for fname in sorted(os.listdir(val_image_dir)):
    # depth map
    if fname == '1_1_22_3':
        continue
    video_type = int(fname.split('_')[-1])
    l = 1 if video_type == 1 else 0
    labels.append(l)


# In[47]:


# finetune 
best_model = None
best_aou = 0.0

for epoch in range(num_epochs):
    print("Epoch {} / {}".format(epoch, num_epochs))
    model.train()
    scheduler.step()
    train_one_epoch()
    
    # eval every epochs
    model.eval()
    avg_acc, score_list = eval_one_epoch()
    print(score_list[:10])

    aou = metrics.roc_auc_score(labels, score_list)
    print("Eval average acc: {}, aou: {}".format(avg_acc, aou))

    if aou >= best_aou:
        best_aou = aou
        best_model = copy.deepcopy(model.state_dict())

    if epoch % 10 == 0:
        torch.save(best_model, finetune_model_path)
        print("model saved")
    
#     if epoch > 3:
#         break


# In[ ]:




