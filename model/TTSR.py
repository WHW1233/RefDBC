from model import MainNet, LTE, Transfer, ComputeS, Compress, Downsample
from model.GMM_com.model_GMM import *

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import cv2

# jpeg model
from model.DiffJPEG.DiffJPEG import DiffJPEG
import matplotlib.pyplot as plt

class TTSR(nn.Module):
    def __init__(self, args, h=256//2, w=256//2, q=70):
        super(TTSR, self).__init__()
        self.args = args
        self.is_eval = args.eval
        self.is_test = args.test
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )
        self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, 
            res_scale=args.res_scale)
        self.LTE      = LTE.LTE(requires_grad=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss
        # self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.Transfer = Transfer.Transfer()
        # self.Downsample = Downsample.Downsample()
        out_channel_N = 192 #192 or 256
        self.Compress = ImageCompressor(out_channel_N)
        self.Compress2 = ImageCompressor(256)
        self.ComputeS = ComputeS.ComputeS()
        self.Compress_jpeg = Compress.Compress_jpeg()
        self.Compress_bpg = Compress.Compress_bpg()
        # self.Compress_normal = Compress.Compress_normal()
        self.Compress_SH = Compress.Compress_SH()
        self.Compress00 = Compress.Compress00
        self.jpeg = DiffJPEG(h, w, differentiable=True, quality=q)

    def forward(self, LR = None, lrsr=None, ref=None, refsr=None, hr=None, sr=None, idx=0):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3
        t1 = time.time()
        _, lrsr_lv2, _ = self.LTE((lrsr.detach() + 1.) / 2.)
        hr_lv1, hr_lv2, _ = self.LTE((hr.detach() + 1.) / 2.)
        # _, refsr_lv2, _ = self.LTE((refsr.detach() + 1.) / 2.)
        ref_lv1, ref_lv2, _ = self.LTE((ref.detach() + 1.) / 2.)

        S, H, command, Point = self.ComputeS(hr_lv2,ref_lv2)
        t2 = time.time()
        # with open('G:/WHW/compute_time/ComputeS.txt', 'a+') as f:
        #     f.write(str(t2-t1)+'\n')
        S_old = S
        lr = LR
        bpp = 0
        mse_loss = 0

        # keshihua 1.20
        # S_ref = Point[0,0,:].detach().cpu().numpy()
        # S_ref = ((S_ref-0.05)*1.2-0.2) * 250
        # h,w = S_ref.shape[:2]
        # for n_i in range(h):
        #     for n_j in range(w):
        #         if S_ref[n_i, n_j] < 150:
        #             S_ref[n_i, n_j] -= 10
        #         elif S_ref[n_i, n_j] < 200:
        #             S_ref[n_i, n_j] -= 5
        # S_ref = S_ref.astype(np.uint8)
        # S_color = cv2.applyColorMap(S_ref, cv2.COLORMAP_JET)
        # cv2.imwrite('./compression_data/ref_matrix/base5_ref_sim/{}.png'.format(idx), S_color)
        # Point = []

        if self.is_eval or self.is_test:
            print('is evaling')
            # hr , bits = self.Compress_jpeg(hr, idx=idx, rate=16)
            # lr, bits = self.Compress_normal(lr, idx=idx, level=0)
            if command is '95':
                S, H, _, _ = self.ComputeS(lrsr_lv2, ref_lv2)

            # 导入GMM模型
            # if idx == 0:
            #     f = r'G:\WHW\研究生项目\压缩代码\pretrained\GMM_pretrained\pretrain_model_512.pth.tar'
            #     load_model(self.Compress, f)
            #     print("loading success")
            # lr = (lr + 1.) / 2
            #
            # clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = self.Compress(lr)
            # lr = clipped_recon_image * 2. - 1.
            # bits = bpp * lr.size(-2) * lr.size(-1) / 8

            # 使用hevc-intra压缩,q调整压缩图像质量
            # lr, bits = self.Compress_bpg(lr, idx=idx, q=33 )    # hevc compression

            # 使用jpeg2000压缩,rate调整压缩倍数
            lr, bits = self.Compress_jpeg(lr, idx=idx,rate=50)          # jpeg 2000

            # 使用jpeg压缩,在初始化时用quality调整压缩图像质量
            # clipped_recon_image = self.jpeg(lr)                 $ jpeg
            # lr = clipped_recon_image * 2. - 1.
            # bits = bpp * lr.size(-2) * lr.size(-1) / 8

            if command is '90':
                S, R_lv2_star_arg, ref, bits_other, ref_lv1, ref_lv2 = self.Compress_SH(S, H, ref, hr, ref_lv1, ref_lv2, hr_lv1, hr_lv2, Point, idx=idx)
                bits += bits_other

            if command is '70':
                S, H, _, _ = self.ComputeS(lrsr_lv2, ref_lv2)       # test 1.21
                S, R_lv2_star_arg, ref, bits_other, ref_lv1, ref_lv2 = self.Compress_SH(S, H, ref, hr, ref_lv1, ref_lv2, hr_lv1, hr_lv2, Point, idx=idx)
                bits += bits_other
                ref_lv1, ref_lv2, _ = self.LTE((ref.detach() + 1.) / 2.)        # test 1.21
            bpp = bits/hr.size(-2)/hr.size(-1)*8


        else:
            GMM_com = True     #没有使用GMM压缩，就会使用jpeg压缩
            if GMM_com:
                # GMM compress
                self.Compress.train()
                if idx == 0:
                    f = r'./pretrained/GMM_pretrained/pretrain_model_512.pth.tar'
                    load_model(self.Compress, f)
                lr = (lr+1.)/2
                clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = self.Compress(lr)
                lr = clipped_recon_image*2.-1.
            else:
                lr = (lr+1.)/2.
                recon = self.jpeg(lr)
                lr = recon*2. - 1.


        H = H.reshape(H.size(0), H.size(1), H.size(2) * H.size(3))
        T_lv2, T_lv1 = self.Transfer(H, lrsr_lv2, ref_lv1, ref_lv2)


        if command is not '00':
            sr = self.MainNet(lr, S, T_lv2, T_lv1)
        return sr, S_old, lr, T_lv2, T_lv1, Point, command, mse_loss, bpp


