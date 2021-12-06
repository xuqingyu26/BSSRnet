from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import numpy as np
from skimage import measure
from torch.nn import init
import torch
import torch.nn as nn
from torch.nn import Softmax

#####配置训练、验证、测试数据集，声明要使用的一些基础函数

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = os.listdir(self.dataset_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')
        img_hr_left = np.array(img_hr_left, dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left = np.array(img_lr_left, dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)

        img_hr_left, img_hr_right, img_lr_left, img_lr_right = augumentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)
    def __len__(self):
        return len(self.file_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor
        self.file_list = os.listdir(dataset_dir + '/hr')
    def __getitem__(self, index):
        hr_image_left  = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr0.png')
        hr_image_right = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr1.png')
        lr_image_left  = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' +
                                    self.file_list[index] + '/lr0.png')
        lr_image_right = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' +
                                    self.file_list[index] + '/lr1.png')
        hr_image_left  = ToTensor()(hr_image_left)
        hr_image_right = ToTensor()(hr_image_right)
        lr_image_left  = ToTensor()(lr_image_left)
        lr_image_right = ToTensor()(lr_image_right)
        return hr_image_left, hr_image_right, lr_image_left, lr_image_right
    def __len__(self):
        return len(self.file_list)

def augumentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
    if random.random()<0.5:
        lr_image_left_ = lr_image_right[:, ::-1, :]   #H*W*C,W维度逆序排列，相当于图形水平翻转
        lr_image_right_ = lr_image_left[:, ::-1, :]
        hr_image_left_ = hr_image_right[:, ::-1, :]
        hr_image_right_ = hr_image_left[:, ::-1, :]
        lr_image_left, lr_image_right = lr_image_left_, lr_image_right_
        hr_image_left, hr_image_right = hr_image_left_, hr_image_right_
    if random.random()<0.5:
        lr_image_left = lr_image_left[::-1, :, :]  # H*W*C,H维度逆序排列，相当于图形垂直翻转
        lr_image_right = lr_image_right[::-1, :, :]
        hr_image_left = hr_image_left[::-1, :, :]
        hr_image_right = hr_image_right[::-1, :, :]
    return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right),\
           np.ascontiguousarray(lr_image_left),np.ascontiguousarray(lr_image_right)        #返回array占用内存为连续值，加速gpu运行速度

def toTensor(img):
    img = torch.from_numpy(img.transpose((2,0,1)))  #H*W*C to C*H*W
    return img.float().div(255)    #数值归一化

class L1Loss(object):
    def __call__(self, input, target):
        return torch.abs(input - target).mean()

def cal_psnr(img1, img2):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    return measure.compare_psnr(img1_np, img2_np)

def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)      # All convs initiated by Xavier

class L1_Charbonnier_loss(nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff*diff + self.eps)
        loss = torch.mean(error)
        return loss

def space_to_depth(img, r):
    B, C , H, W = img.shape
    h = int(H/r)
    w = int(W/r)
    img = img.view(B, C, h, r, w, r).permute(0, 1, 3, 5, 2, 4).contiguous()
    img = img.view(B, C*r**2, h, w)
    return img


def Disparity(Q, K):
    B, C, H, W = Q.shape
    score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, W, C),
                      K.permute(0, 2, 1, 3).contiguous().view(-1, C, W))
    disparity_right = torch.argmax(score, dim=-1).view(B, H, W)
    disparity_left = torch.argmax(score.permute(0, 2, 1), dim=-1).view(B, H, W)
    I = torch.arange(W).repeat(B, H, 1)
    Disparity_left = I - disparity_left
    Disparity_right = I -disparity_right
    return Disparity_left.float(), Disparity_right.float()