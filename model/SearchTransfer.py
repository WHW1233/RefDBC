import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from imageio import imread
from imageio import imwrite
from PIL import Image
from compression_data import H_conpress


class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

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


    def forward(self, lrsr_lv1, lrsr_lv2, lrsr_lv3, ref_lv1, ref_lv2, ref_lv3, idx=0):
        ### search
        refsr_lv2 = ref_lv2
        lrsr_lv2_unfold  = F.unfold(lrsr_lv2, kernel_size=(3, 3), padding=1)
        refsr_lv2_unfold = F.unfold(refsr_lv2, kernel_size=(3, 3), padding=1)
        refsr_lv2_unfold = refsr_lv2_unfold.permute(0, 2, 1)

        refsr_lv2_unfold = F.normalize(refsr_lv2_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv2_unfold  = F.normalize(lrsr_lv2_unfold, dim=1) # [N, C*k*k, H*W]

        R_lv2 = torch.bmm(refsr_lv2_unfold, lrsr_lv2_unfold) #[N, Hr*Wr, H*W]
        R_lv2_star, R_lv2_star_arg = torch.max(R_lv2, dim=1) #[N, H*W]

        # encode H
        H = R_lv2_star_arg.cpu().numpy()
        # np.savez_compressed('./compression_data/encode/H_{}'.format(idx), H)
        # H = np.load('./compression_data/encode/H_{}.npz'.format(idx))
        # H = H['arr_0']
        H = H_conpress.H_compress(H, h=128,w=128,type='x4', idx=idx)
        R_lv2_star_arg = torch.from_numpy(H).cuda()

        S = R_lv2_star.view(R_lv2_star.size(0), 1, lrsr_lv2.size(2), lrsr_lv2.size(3))
        H = R_lv2_star_arg.view(R_lv2_star_arg.size(0), 1, lrsr_lv2.size(2), lrsr_lv2.size(3))
        L = [i for i in range(lrsr_lv2.size(2)*lrsr_lv2.size(3))]
        X = torch.LongTensor(L)
        X = X.view(R_lv2_star.size(0), 1, lrsr_lv2.size(2), lrsr_lv2.size(3)).cuda()
        N1=lrsr_lv2.size(2)//32
        N2=lrsr_lv2.size(3)//32
        Point = []
        # for i_h in range(N1):
        #     for i_w in range(N2):
        #         if torch.mean(S[:,:,i_h*32:i_h*32+32, i_w*32:i_w*32+32])<0.9:
        #             Point.append((i_h,i_w))
        #             S[:, :, i_h * 32:i_h * 32 + 32, i_w * 32:i_w * 32 + 32] = 1
        #             H[:, :, i_h * 32:i_h * 32 + 32, i_w * 32:i_w * 32 + 32] = X[:, :, i_h * 32:i_h * 32 + 32, i_w * 32:i_w * 32 + 32]
        #             ref_lv1[:,:,i_h * 64:i_h * 64 + 64, i_w * 64:i_w * 64 + 64] = lrsr_lv1[:,:,i_h * 64:i_h * 64 + 64, i_w * 64:i_w * 64 + 64]
        #             ref_lv2[:,:,i_h * 32:i_h * 32 + 32, i_w * 32:i_w * 32 + 32] = lrsr_lv2[:,:,i_h * 32:i_h * 32 + 32, i_w * 32:i_w * 32 + 32]
        R_lv2_star_arg = H.view(R_lv2_star.size(0), 1, lrsr_lv2.size(2)*lrsr_lv2.size(3))
        ### transfer

        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(3, 3), padding=1)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(6, 6), padding=2, stride=2)


        # T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv2_star_arg)
        # print(R_lv2_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv2_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv2_star_arg)

        # T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv2.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv2.size(2), lrsr_lv2.size(3)), kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv2.size(2)*2, lrsr_lv2.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        # temp = 0.85*torch.ones(R_lv2_star.size(0), 1, lrsr_lv2.size(2), lrsr_lv2.size(3)).cuda()
        # S = torch.maximum(S, temp)
        S_save = S * 250
        # print(S_save.cpu().numpy().dtype)
        S_save = np.uint8(S_save[0][0].cpu().numpy())
        # encode
        im = Image.fromarray(S_save)
        im.save('./compression_data/S/S_{}.png'.format(idx))
        S_save = imread('./compression_data/S/S_{}.png'.format(idx))
        root = 'D:/文件DA/科研/视频编码/代码/jpeglib/openjpeg-v2.3.0-windows-x64/openjpeg-v2.3.0-windows-x64/bin'
        cmd1 = '{}/opj_compress.exe -i ./compression_data/S/S_{}.png \
                                -o ./compression_data/encode/lr_S_{}.J2K -r {}'.format(root, idx, idx, 20)
        os.system(cmd1)
        # decode
        cmd2 = '{}/opj_decompress.exe -i ./compression_data/encode/lr_S_{}.J2K -o new.png'.format(root, idx)
        os.system(cmd2)
        S_com = imread('new.png')
        S_com = np.float32(S_com)
        S_save = S_com
        S[0][0] = torch.from_numpy(S_save / 250).cuda()
        os.remove('new.png')

        return S, R_lv2_star_arg, T_lv2, T_lv1, Point