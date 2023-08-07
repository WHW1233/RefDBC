import os
from imageio import imread
from imageio import imwrite
from PIL import Image
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 as cv
# Ignore warnings
import warnings

# degradation
from dataset import degradation
import math
import time

DOWN_SCALE = 2
warnings.filterwarnings("ignore")
SCALE = 128
class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()  # rot90,矩阵逆时针旋转90*k1
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        LR, LR_bic, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_bic'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2, 0, 1))
        LR_bic = LR_bic.transpose((2, 0, 1))
        LR_sr = LR_sr.transpose((2, 0, 1))
        HR = HR.transpose((2, 0, 1))
        Ref = Ref.transpose((2, 0, 1))
        Ref_sr = Ref_sr.transpose((2, 0, 1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_bic': torch.from_numpy(LR_bic).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])):
        self.input_list = sorted([os.path.join(args.dataset_dir, 'train/input', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, 'train/input'))])
        self.input_list = self.input_list[:100]
        self.ref_list = sorted([os.path.join(args.dataset_dir, 'train/ref', name) for name in
                                os.listdir(os.path.join(args.dataset_dir, 'train/ref'))])
        self.ref_list = self.ref_list[:100]
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.input_list[idx])
        h, w = HR.shape[:2]  # 获取高和宽
        # HR = HR[:h//4*4, :w//4*4, :]
        if h>SCALE:
            HR = HR[:SCALE,:SCALE,:]
            h,w = SCALE, SCALE

        # blur
        kernel_size = 7
        sigma = 0.5
        kernel = degradation.bivariate_Gaussian(kernel_size, sigma, 5, -math.pi, isotropic=True)
        hr_img = cv.filter2D(HR, -1, kernel=kernel)


        ### LR and LR_sr
        # hr_img = cv.GaussianBlur(HR, (0, 0), 1, 1)
        # LR_path = self.input_list[idx].replace('input', 'LR')
        # LR = imread(LR_path)
        LR_bic = np.array(Image.fromarray(hr_img).resize((w // DOWN_SCALE, h // DOWN_SCALE), Image.BICUBIC))  # 4倍降采样
        LR_sr = np.array(Image.fromarray(LR_bic).resize((w, h), Image.BICUBIC))  # 4倍上采用
        LR = np.array(Image.fromarray(hr_img))

        ### Ref and Ref_sr
        Ref_sub = imread(self.ref_list[idx])
        h2, w2 = Ref_sub.shape[:2]
        if h2>SCALE:
            Ref_sub = Ref_sub[:SCALE, :SCALE, :]
            h2, w2 = SCALE, SCALE
        Ref_sr_sub = np.array(Image.fromarray(Ref_sub).resize((w2 // DOWN_SCALE, h2 // DOWN_SCALE), Image.BICUBIC))
        Ref_sr_sub = np.array(Image.fromarray(Ref_sr_sub).resize((w2, h2), Image.BICUBIC))  # 4倍降采样再上采样

        ### complete ref and ref_sr to the same size, to use batch_size > 1
        # Ref = np.zeros((SCALE, SCALE, 3))
        # Ref_sr = np.zeros((SCALE, SCALE, 3))
        # Ref[:h2, :w2, :] = Ref_sub
        # Ref_sr[:h2, :w2, :] = Ref_sr_sub  # 把ref尺度大小和原图像大小变为一致

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        LR_bic = LR_bic.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref_sub.astype(np.float32)
        Ref_sr = Ref_sr_sub.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        LR_bic = LR_bic/127.5 -1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,
                  'LR_sr': LR_sr,
                  'LR_bic':LR_bic,
                  'HR': HR,
                  'Ref': Ref,
                  'Ref_sr': Ref_sr
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        # self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', '*_0.tif')))
        # self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5',
        #                                               '*_' + ref_level + '.tif')))
        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', '*_0.tif')))
        # ref_level = '6'
        self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5',
                                                      '*_' + ref_level + '.tif')))
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.input_list[idx])
        # HR = HR[400:400+256, 600:600+256, :]        # oxford car
        h, w = HR.shape[:2]
        h, w = h // DOWN_SCALE * DOWN_SCALE, w // DOWN_SCALE * DOWN_SCALE
        HR = HR[:h, :w, :]  ### crop to the multiple of 4
        # f = open('table.csv', 'a+', encoding='utf-8', newline='')
        # csv_writer = csv.writer(f)
        name = self.input_list[idx].split('\\')[-1]
        # csv_writer.writerow([name, idx])

        # blur
        kernel_size = 7
        sigma = 1
        kernel = degradation.bivariate_Gaussian(kernel_size, sigma, 5, -math.pi, isotropic=True)
        hr_img = cv.filter2D(HR, -1, kernel=kernel)
        # hr_img = HR


        ### LR and LR_sr
        # hr_img = cv.GaussianBlur(HR, (0, 0), 1, 1)
        LR_bic = np.array(Image.fromarray(hr_img).resize((w // DOWN_SCALE, h // DOWN_SCALE), Image.BICUBIC))  # 4倍降采样
        LR_sr = np.array(Image.fromarray(LR_bic).resize((w, h), Image.BICUBIC))  # 4倍上采用
        LR = np.array(Image.fromarray(hr_img))

        ### Ref and Ref_sr
        Ref = imread(self.ref_list[idx])
        # Ref = Ref[400:400 + 256, 600:600 + 256, :]  # oxford car
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2 // DOWN_SCALE * DOWN_SCALE, w2 // DOWN_SCALE * DOWN_SCALE
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2 // DOWN_SCALE, h2 // DOWN_SCALE), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        LR_bic = LR_bic.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        LR_bic = LR_bic/127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,
                  'LR_sr': LR_sr,
                  'LR_bic':LR_bic,
                  'HR': HR,
                  'Ref': Ref,
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample