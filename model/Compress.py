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
from compression_data import H_conpress
import time
import sys
sys.path.append('D:/文件DA/科研/视频编码/代码/GMM_image_compression')
import GMM_compression

# level 0-5
lr_com_level = 0
S_rate = 60
S_q = 30
hr_com_rate = 100


class Compress_jpeg(nn.Module):
    def __init__(self):
        super(Compress_jpeg, self).__init__()

    def forward(self, lr, idx=0, rate=16):
        ### search
        LR =  (lr + 1.) * 127.5
        LR_save = np.transpose(LR.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        imwrite('./compression_data/lr_image/lr_{}.tif'.format(idx), LR_save)  # added by whw
        # encode LR
        root = 'D:/文件DA/科研/视频编码/代码/jpeglib/openjpeg-v2.3.0-windows-x64/openjpeg-v2.3.0-windows-x64/bin'
        cmd1 = '{}/opj_compress.exe -i ./compression_data/lr_image/lr_{}.tif \
                                -o ./compression_data/encode/lr_{}.J2K -r {}'.format(root, idx, idx, rate)
        time1 = time.time()
        os.system(cmd1)
        time2 = time.time()
        # with open('G:/WHW/compute_time/our_encode_time.txt', 'a+') as f:
        #     f.write(str(time2 - time1)+'\n')

        bits = os.path.getsize('./compression_data/encode/lr_{}.J2K'.format(idx))
        # decode LR
        cmd2 = '{}/opj_decompress.exe -i ./compression_data/encode/lr_{}.J2K -o new.tif'.format(root, idx)
        time1 = time.time()
        os.system(cmd2)
        time2 = time.time()
        # with open('G:/WHW/decode_time.txt', 'a') as f:
        #     f.write(str(time2 - time1)+'\n')
        LR = imread('new.tif')
        LR = LR.swapaxes(0, 2)
        LR = LR.swapaxes(1, 2)
        LR = LR.reshape(1, 3, lr.size(2), lr.size(3))
        os.remove('new.tif')
        lr = LR.astype(np.float32)
        lr = lr / 127.5 - 1.
        lr = torch.from_numpy(lr).cuda()

        return lr, bits

class Compress_bpg(nn.Module):
    def __init__(self):
        super(Compress_bpg, self).__init__()

    def forward(self, lr, idx=0, q=16):
        ### search
        LR =  (lr + 1.) * 127.5
        LR_save = np.transpose(LR.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        imwrite('./compression_data/lr_image/lr_{}.png'.format(idx), LR_save)  # added by whw
        # encode LR
        root = 'D:/文件DA/科研/视频编码/代码/bpg'
        cmd1 = '{}/bpgenc.exe -m 9 -b 8 -q {} ./compression_data/lr_image/lr_{}.png \
        -o ./compression_data/encode/lr_{}.bpg '.format(root, q, idx, idx)
        os.system(cmd1)
        bits = os.path.getsize('./compression_data/encode/lr_{}.bpg'.format(idx))
        # decode LR
        cmd2 = '{}/bpgdec.exe -o new.png ./compression_data/encode/lr_{}.bpg'.format(root, idx)
        os.system(cmd2)
        LR = imread('new.png')
        LR = LR.swapaxes(0, 2)
        LR = LR.swapaxes(1, 2)
        LR = LR.reshape(1, 3, lr.size(2), lr.size(3))
        os.remove('new.png')
        lr = LR.astype(np.float32)
        lr = lr / 127.5 - 1.
        lr = torch.from_numpy(lr).cuda()

        return lr, bits


class Compress_normal(nn.Module):
    def __init__(self):
        super(Compress_normal, self).__init__()

    def forward(self, lr, idx=0, level=3):
        ### search
        LR =  (lr + 1.) * 127.5
        LR_save = np.transpose(LR.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        imwrite('./compression_data/lr_image/lr_{}.tif'.format(idx), LR_save)  # added by whw
        # encode LR
        LR = Image.open('./compression_data/lr_image/lr_{}.tif'.format(idx)).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        LR = transform(LR)
        LR = LR.reshape((1, LR.size(0), LR.size(1), LR.size(2)))
        model_path = 'D:/文件DA/科研/视频编码/代码/GMM_image_compression/pretrained_model'
        lr, bits_lr, bpp, psnr = GMM_compression.use_GMM_com(LR, pretrain_path=model_path, com_level=level, idx=idx)
        # lr = torch.from_numpy(lr).cuda()
        # lr = lr.unsqueeze(0)

        bits = bits_lr
        return lr, bits



class Compress_SH(nn.Module):
    def __init__(self):
        super(Compress_SH, self).__init__()

    def save_patch(self,HR_save, ref_save, ref_lv1,ref_lv2,hr_lv1,hr_lv2,Point, idx):
        bits = 0
        scale1 = hr_lv1.size(2) // 4
        scale2 = scale1//2
        hard_encode = 0
        hard_decode = 0
        for P in Point:
            HR_patch = HR_save[P[0] * scale1:P[0] * scale1 + scale1, P[1] * scale1:P[1] * scale1 + scale1, :]
            ref_save[P[0] * scale1:P[0] * scale1 + scale1, P[1] * scale1:P[1] * scale1 + scale1, :] = HR_patch
            HR_patch_filename = 'hr_{}_patch_{}_{}'.format(idx, P[0], P[1])
            imwrite('./compression_data/lr_image/{}.tif'.format(HR_patch_filename), HR_patch)  # added by whw
            root = 'D:/文件DA/科研/视频编码/代码/jpeglib/openjpeg-v2.3.0-windows-x64/openjpeg-v2.3.0-windows-x64/bin'
            cmd1 = '{}/opj_compress.exe -i ./compression_data/lr_image/{}.tif \
                        -o ./compression_data/encode/{}.J2K -r {}'.format(root, HR_patch_filename, HR_patch_filename,
                                                                          hr_com_rate)
            t1 = time.time()
            os.system(cmd1)
            t2 = time.time()
            hard_encode += t2-t1
            bits += os.path.getsize('./compression_data/encode/{}.J2K'.format(HR_patch_filename))
            ref_lv1[:, :, P[0] * scale1:P[0] * scale1 + scale1, P[1] * scale1:P[1] * scale1 + scale1] = \
                hr_lv1[:, :, P[0] * scale1:P[0] * scale1 + scale1, P[1] * scale1:P[1] * scale1 + scale1]
            ref_lv2[:, :, P[0] * scale2:P[0] * scale2 + scale2, P[1] * scale2:P[1] * scale2 + scale2] = \
                hr_lv2[:, :, P[0] * scale2:P[0] * scale2 + scale2, P[1] * scale2:P[1] * scale2 + scale2]
        # with open('G:/WHW/compute_time/hard_compress.txt', 'a') as f:
        #     f.write(str(hard_encode)+'\n')
        return ref_save, bits


    def forward(self, S, H, ref, hr,  ref_lv1, ref_lv2, hr_lv1,hr_lv2, Point, idx=0):
        bits = 0

        R_lv2_star_arg = H
        ref = (ref + 1.) * 127.5
        ref_save = np.transpose(ref.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        HR = (hr + 1.) * 127.5
        HR_save = np.transpose(HR.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)

        # encode HR patch
        ref_save, bits_patch = self.save_patch( HR_save, ref_save, ref_lv1,ref_lv2,hr_lv1,hr_lv2,Point, idx)
        ref_save = ref_save.swapaxes(0, 2)
        ref_save = ref_save.swapaxes(1, 2)
        ref_save = ref_save.astype(np.float32)
        ref = torch.from_numpy(ref_save/127.5-1).cuda()
        ref = ref.unsqueeze(0)
        bits += bits_patch

        # mytest 1.21
        # '''
        scale = hr_lv2.size(2)//4
        L = [i for i in range(hr_lv2.size(2) * hr_lv2.size(3))]
        X = torch.LongTensor(L)
        X = X.view(hr_lv2.size(0), 1, hr_lv2.size(2), hr_lv2.size(3)).cuda()
        for P in Point:
            S[:, :, P[0] * scale:P[0] * scale + scale, P[1] * scale:P[1] * scale + scale] = 0.96
            H[:, :, P[0] * scale:P[0] * scale + scale, P[1] * scale:P[1] * scale + scale] = \
                X[:, :, P[0] * scale:P[0] * scale + scale, P[1] * scale:P[1] * scale + scale]

        if len(Point) == 0:
            # encode S
            S_save = S * 250
            S_save = np.uint8(S_save[0][0].cpu().numpy())
            im = Image.fromarray(S_save)
            im.save('./compression_data/S/S_{}.png'.format(idx))
            S_save = imread('./compression_data/S/S_{}.png'.format(idx))
            root = 'D:/文件DA/科研/视频编码/代码/jpeglib/openjpeg-v2.3.0-windows-x64/openjpeg-v2.3.0-windows-x64/bin'
            cmd1 = '{}/opj_compress.exe -i ./compression_data/S/S_{}.png \
                                                -o ./compression_data/encode/lr_S_{}.J2K -r {}'.format(root, idx, idx,
                                                                                                       S_rate)

            os.system(cmd1)
            bits += os.path.getsize('./compression_data/encode/lr_S_{}.J2K'.format(idx))
            # decode S
            cmd2 = '{}/opj_decompress.exe -i ./compression_data/encode/lr_S_{}.J2K -o new.png'.format(root, idx)
            os.system(cmd2)
            S_com = imread('new.png')
            S_com = np.float32(S_com)
            S_save = S_com
            S[0][0] = torch.from_numpy(S_save / 250).cuda()
            os.remove('new.png')


            # encode H
            R_lv2_star_arg = H.view(R_lv2_star_arg.size(0), 1, hr_lv2.size(2) * hr_lv2.size(3))
            H = R_lv2_star_arg.cpu().numpy()
            # H, bits_h = H_conpress.H_compress(H, h=128, w=128, type='x2', idx=idx)
            R_lv2_star_arg = torch.from_numpy(H).cuda()
            bits += 128*128*(1.25)/16


        return S, R_lv2_star_arg, ref, bits, ref_lv1, ref_lv2

def Compress00( lr, hr,idx=0):
    # encode HR
    LR = (lr + 1.) * 127.5
    LR_save = np.transpose(LR.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    HR = (hr + 1.) * 127.5
    HR_save = np.transpose(HR.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    imwrite('./compression_data/lr_image/lr_{}.tif'.format(idx), LR_save)  # added by whw
    imwrite('./compression_data/lr_image/lr_{}_hr.tif'.format(idx), HR_save)  # added by whw

    HR = Image.open('./compression_data/lr_image/lr_{}_hr.tif'.format(idx)).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    HR = transform(HR)
    HR = HR.reshape((1, HR.size(0), HR.size(1), HR.size(2)))
    # HR /=2    #辐射降采样
    model_path = 'D:/文件DA/科研/视频编码/代码/GMM_image_compression/pretrained_model'
    HR, bits, bpp, psnr = GMM_compression.use_GMM_com(HR, pretrain_path=model_path, com_level=lr_com_level, idx=idx, is_H=True)
    hr = HR * 4 - 1.

    return hr, bits

if __name__ == '__main__':
    pass
