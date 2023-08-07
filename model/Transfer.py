import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
from imageio import imread
from imageio import imwrite
from PIL import Image


class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()

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


    def forward(self, R_lv2_star_arg, lrsr_lv2, ref_lv1, ref_lv2):
        ### transfer
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(3, 3), padding=1)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(6, 6), padding=2, stride=2)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv2_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv2_star_arg)

        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv2.size(2), lrsr_lv2.size(3)), kernel_size=(3, 3),
                       padding=1) / (3. * 3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv2.size(2) * 2, lrsr_lv2.size(3) * 2), kernel_size=(6, 6),
                       padding=2, stride=2) / (3. * 3.)

        return T_lv2, T_lv1