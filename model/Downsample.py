import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import MeanShift
import numpy as np
from imageio import imread, imsave
import os



class Downsample(torch.nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.Conv1 = nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0)
        self.Conv2 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.Conv3 = nn.Conv2d(6,3,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()


    def get_res(self, hr, ref):
        return hr - ref

    def forward(self, hr, ref, LR):
        res = self.get_res(hr, ref)
        res = self.Conv1(res)
        res = self.relu(res)

        lr = self.Conv2(LR)
        lr = self.Conv3(torch.cat((lr,res), dim=1))
        lr = torch.clamp(lr,-1, 1)
        lr_save = (lr + 1.) * 127.5
        lr_save = np.transpose(lr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        imsave('直接得到的lr.png', lr_save)
        return lr


if __name__=='__main__':
    K = Downsample()
    HR = torch.randn(1,3,256,256)
    ref = HR + 3
    lr = K(HR, ref, HR)
    print(lr.shape)
    print(lr)