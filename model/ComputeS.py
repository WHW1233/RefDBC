import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from imageio import imread
from imageio import imwrite
from PIL import Image


class ComputeS(nn.Module):
    def __init__(self):
        super(ComputeS, self).__init__()

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        # print(index)
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


    def forward(self, lrsr_lv2,  ref_lv2 ):
        ### search
        refsr_lv2 = ref_lv2
        lrsr_lv2_unfold  = F.unfold(lrsr_lv2, kernel_size=(3, 3), padding=1)
        refsr_lv2_unfold = F.unfold(refsr_lv2, kernel_size=(3, 3), padding=1)
        refsr_lv2_unfold = refsr_lv2_unfold.permute(0, 2, 1)
        refsr_lv2_unfold = F.normalize(refsr_lv2_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv2_unfold  = F.normalize(lrsr_lv2_unfold, dim=1) # [N, C*k*k, H*W]


        R_lv2 = torch.bmm(refsr_lv2_unfold, lrsr_lv2_unfold) #[N, Hr*Wr, H*W]
        R_lv2_star, R_lv2_star_arg = torch.max(R_lv2, dim=1) #[N, H*W]
        # R_lv2_star *= 0.98
        # ref xiang guan xi shu
        cal_ref = False
        if cal_ref:
            lrsr_ref_unfold = F.unfold(lrsr_lv2, kernel_size=(3, 3), padding=1)
            refsr_ref_unfold = F.unfold(refsr_lv2, kernel_size=(3, 3), padding=1)
            lrsr_ref_unfold = lrsr_ref_unfold.permute(0, 2, 1)
            lrsr_ref_unfold = F.normalize(lrsr_ref_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
            refsr_ref_unfold = F.normalize(refsr_ref_unfold, dim=1)  # [N, C*k*k, H*W]
            R_ref_lv2 = torch.bmm(lrsr_ref_unfold,refsr_ref_unfold)
            R_ref, _ = torch.max(R_ref_lv2, dim=1)  # [N, H*W]
            # R_ref *= 0.98
            S_ref = R_ref.view(R_ref.size(0), 1, lrsr_lv2.size(2), lrsr_lv2.size(3))

        S = R_lv2_star.view(R_lv2_star.size(0), 1, lrsr_lv2.size(2), lrsr_lv2.size(3))
        H = R_lv2_star_arg.view(R_lv2_star_arg.size(0), 1, lrsr_lv2.size(2), lrsr_lv2.size(3))

        # level = [0.86, 0.87, 0]        # compress x2 model   .90 .86   #level1 bpp .2552695224, PSMR 33.003, SSIM 0.8673
        # level = [0.90, 0.93, 0]     #   level2 bpp .34944225, PSNR 34.029, SSIM .9076
        # level = [0.94, 0.95,  0.]     # level3  bpp .40423640200, PSNR 34.204, SSIM .9177
        # level = [0.97, 0.95, 0.]        # with bpg robotcar
        # level = [0.90, 12, 0.]        # gmm adaptive
        # level = [0.90, 0.88, 0.]        #计算复杂度用的参数
        # level = [0, 12,13]
        level = [0,10,11]           # 训练时调整此值为0，10，11，测试时调整为0.9，0.88，0
        scale = lrsr_lv2.size(2) // 4
        N1=lrsr_lv2.size(2)//scale
        N2=lrsr_lv2.size(3)//scale
        Point = []
        K = torch.mean(S)
        if level[0] is 0 or K >= level[0]:
            command = '95'
            Point = None
            return S, H, command, Point
        elif K >=level[1]:
            command = '90'
            Point = []
            if cal_ref:
                Point = S_ref
            return S, H, command, Point
        elif K >=level[2]:
            command = '70'
            L = [i for i in range(lrsr_lv2.size(2) * lrsr_lv2.size(3))]
            X = torch.LongTensor(L)
            X = X.view(R_lv2_star.size(0), 1, lrsr_lv2.size(2), lrsr_lv2.size(3)).cuda()
            for i_h in range(N1):
                for i_w in range(N2):
                    if torch.mean(S[:, :, i_h * scale:i_h * scale + scale, i_w * scale:i_w * scale + scale]) < 0.92: #0.90 :#92:
                        Point.append((i_h, i_w))
                        # S[:, :, i_h * scale:i_h * scale + scale, i_w * scale:i_w * scale + scale] = 0.99
                        # H[:, :, i_h * scale:i_h * scale + scale, i_w * scale:i_w * scale + scale] = \
                        #     X[:, :, i_h * scale:i_h * scale + scale, i_w * scale:i_w * scale + scale]
            return S, H, command, Point
        else:
            command = '00'
            Point = None
            return S, H, command, Point