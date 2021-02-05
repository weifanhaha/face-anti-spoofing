# Face Anti-Spoofing with CDCN model

Implemented [Central Difference Convolutional Networks](https://arxiv.org/pdf/2003.04092v1.pdf) for face anti-spoofing task. In this repo, [triplet loss](https://arxiv.org/pdf/1503.03832.pdf) is added to improve the performance of CDCN model.

## Description

### Face Anti-Spoofing

The goal of face anti-spoofing task is to let model distinguish the read images(video), which are face images directly captured by cameras, from fake images(video), which are remade from printed photos, replay-videos, etc.

![image](https://github.com/weifanhaha/face-anti-spoofing/blob/master/images/face_anti_spoofing.png)

### Central Difference Convolutional Networks (CDCN)

CDCN is one of the state-of-the-art models for face anti-spoofing task. It uses a special convolution - Central Difference Convolution (CDC) to capture the invariant detailed spoofing features.

![image](https://i.imgur.com/SJ2CcOC.png)
![image](https://i.imgur.com/fnxMTLX.png)

Reference: [Searching Central Difference Convolutional Networks for Face Anti-Spoofing](https://arxiv.org/pdf/2003.04092v1.pdf)

### Triplet Loss

Triplet loss describes the distance between one training point in feature domain and another one with the same label and also the distance between that training point in feature domain and another one with different label. By adding triplet loss in the training process, we hope that the model learns to reduce the distance between data with same label and increase the distance between data with different label.

![image](https://github.com/weifanhaha/face-anti-spoofing/blob/master/images/triplet_loss.png)

## Usage

### Dataset

We trained the model on OULU-NPU dataset and test on [OULU-NPU](https://sites.google.com/site/oulunpudatabase/) and [SiW](http://cvlab.cse.msu.edu/siw-spoof-in-the-wild-database.html) dataset.
Note: In this project we used cropped images in the two dataset which is not public. To reproduce the result, you may need more preprocessing process.

### Install packages

```
pip install -r requirements.txt
```

### Train

To train CDCN model, you may need to generate depth map on your own. Please refer to [PRNet-Depth-Generation](https://github.com/clks-wzz/PRNet-Depth-Generation).

Train CDCN

```
cd cdcn
python train.py
```

Train CDCN with triplet loss

```
cd cdcn
python triplet_train.py
```

### Predict Labels

You can predict the labels of the images in the test dataset with trained models.

Predict with CDCN

```
cd cdcn
python predict.py
```

Predict with CDCN trained with triplet loss

```
cd cdcn
python triplet_predict.py
```

## Analysis

With triplet loss, the real images get blurred , while the fake images get closed to image filled with 0s and they are closed to the images with the same label.

![image](https://github.com/weifanhaha/face-anti-spoofing/blob/master/images/analysis.png)

## Reference

[CDCN](https://github.com/ZitongYu/CDCN)
