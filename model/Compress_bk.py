import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import cv2 as cv
from imageio import imread
from imageio import imwrite
from PIL import Image
import time
from model.GMM_com.model_GMM import *

# level 0-5
lr_com_level = 0
S_com_level = 5
S_rate = 12
hr_com_rate = 40


out_channel_N = 256
out_channel_M = 256
level_list = ['4096_256', '2048_256', '1024_256', '512_192', '256_192', '128_192']

class Com_level():
    def __init__(self, level, root):
        self.level_list = ['4096_256', '2048_256', '1024_256', '512_192', '256_192', '128_192']
        self.level = level
        self.root = root

    def get_level(self):
        scale = self.level_list[self.level].split('_')[0]
        return scale

    def get_model_path(self):
        file_name = 'pretrain_model_'+ str(self.get_level()) + '.pth.tar'
        path_name = os.path.join(self.root, file_name)
        return path_name

    def get_channel(self):
        out_channel = self.level_list[self.level].split('_')[1]
        return int(out_channel)


def test_net(net, input, idx=0, is_H=False):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        name = 'test_image_{}.png'.format(idx)

        clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(input)
        bits = bpp*input.size(-2)*input.size(-1)/8        # 计算出来降采样后的大小(单位字节)

        save_image = clipped_recon_image * 255
        im = np.uint8(save_image[0].cpu().numpy())
        im = im.swapaxes(0, 2)
        im = im.swapaxes(0, 1)
        hr_img = cv.GaussianBlur(im, (0, 0), 1, 1)
        w,h = hr_img.shape[:2]
        LR = np.array(Image.fromarray(hr_img).resize((w// 2, h// 2), Image.BICUBIC))  # 4倍降采样
        LR = LR.swapaxes(0, 2)
        LR = LR.swapaxes(1, 2)
        LR = LR.astype(np.float32)
        LR = LR / 127.5 - 1.

        mse_loss, bpp_feature, bpp_z, bpp = \
            torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
        psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
        sumBpp += bpp
        sumPsnr += psnr
        cnt += 1

        sumBpp /= cnt
        sumPsnr /= cnt
        print("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}".format(sumBpp, sumPsnr))
        if is_H:
            return clipped_recon_image, bits, bpp, psnr
        else:
            return LR, bits, bpp, psnr


# 需要参数 seed不用，name -n，config，pretrain，val测试数据集路径
def use_GMM_com(LR, pretrain_path='./pretrained_model/', com_level=0, idx=0, is_H=True):
    # 默认值0表示4096_256
    L = Com_level(com_level, pretrain_path)
    pretrain_path = L.get_model_path()
    print("image compression val")
    out_channel_N = L.get_channel()
    out_channel_M = L.get_channel()
    model = ImageCompressor(out_channel_N)
    # pdb.set_trace()
    if pretrain_path != '':
        print("loading model:{}".format(pretrain_path))
        global_step = load_model(model, pretrain_path)
    net = model.cuda()
    clipped_recon_image, bits, bpp, psnr = test_net(net, LR, idx=idx, is_H=is_H)
    return clipped_recon_image, bits, bpp, psnr

class Compress_normal(nn.Module):
    def __init__(self):
        super(Compress_normal, self).__init__()

    def forward(self, lr, idx=0, level=3):

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
        lr, bits_lr, bpp, psnr = use_GMM_com(LR, pretrain_path=model_path, com_level=level, idx=idx)
        # bits_lr /= 4
        lr = lr*2-1
        lr = torch.from_numpy(lr).cuda()
        lr = lr.unsqueeze(0)
        bits = bits_lr
        return lr, bits


class Compress70(nn.Module):
    def __init__(self):
        super(Compress70, self).__init__()

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

    def forward(self, S, H, lr, hr, hr_lv1, hr_lv2, ref_lv1, ref_lv2, Point, idx=0):
        ### search
        R_lv2_star_arg = H
        LR = (lr + 1.) * 127.5
        LR_save = np.transpose(LR.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        HR = (hr + 1.) * 127.5
        HR_save = np.transpose(HR.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        imwrite('./compression_data/lr_image/lr_{}.tif'.format(idx), HR_save)  # added by whw

        # encode LR
        HR = Image.open('./compression_data/lr_image/lr_{}.tif'.format(idx)).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        HR = transform(HR)
        HR = HR.reshape((1, HR.size(0), HR.size(1), HR.size(2)))
        model_path = 'D:/文件DA/科研/视频编码/代码/GMM_image_compression/pretrained_model'
        lr, bits_lr, bpp, psnr = GMM_compression.use_GMM_com(HR, pretrain_path=model_path, com_level=lr_com_level, idx=idx)
        bits_lr /= 4
        bits = bits_lr
        lr = torch.from_numpy(lr).cuda()
        lr = lr.unsqueeze(0)

        # encode HR patch
        for P in Point:
            HR_patch = HR_save[P[0] * 64:P[0] * 64 + 64, P[1] * 64:P[1] * 64 + 64, :]
            HR_patch_filename = 'hr_{}_patch_{}_{}'.format(idx, P[0], P[1])
            imwrite('./compression_data/lr_image/{}.tif'.format(HR_patch_filename), HR_patch)  # added by whw
            root = 'D:/文件DA/科研/视频编码/代码/jpeglib/openjpeg-v2.3.0-windows-x64/openjpeg-v2.3.0-windows-x64/bin'
            cmd1 = '{}/opj_compress.exe -i ./compression_data/lr_image/{}.tif \
                        -o ./compression_data/encode/{}.J2K -r {}'.format(root, HR_patch_filename, HR_patch_filename,
                                                                          hr_com_rate)
            os.system(cmd1)
            bits += os.path.getsize('./compression_data/encode/{}.J2K'.format(HR_patch_filename))
            ref_lv1[:, :, P[0] * 64:P[0] * 64 + 64, P[1] * 64:P[1] * 64 + 64] = \
                hr_lv1[:, :, P[0] * 64:P[0] * 64 + 64, P[1] * 64:P[1] * 64 + 64]
            ref_lv2[:, :, P[0] * 32:P[0] * 32 + 32, P[1] * 32:P[1] * 32 + 32] = \
                hr_lv2[:, :, P[0] * 32:P[0] * 32 + 32, P[1] * 32:P[1] * 32 + 32]

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


        # S_save = S * 250
        # S_save = np.uint8(S_save[0][0].cpu().numpy())
        # im = Image.fromarray(S_save)
        # im.save('./compression_data/S/S_{}.png'.format(idx))
        # S_save = Image.open('./compression_data/S/S_{}.png'.format(idx))
        # S_save = transform(S_save)
        # S_save = S_save.reshape((1, S_save.size(0), S_save.size(1), S_save.size(2)))
        # S_save = S_save.repeat(1, 3, 1, 1)
        # S_save, bits_s, bpsp, psnr = GMM_compression.use_GMM_com(S_save, pretrain_path=model_path, com_level=S_com_level, idx=idx)
        # S_save = S_save[:, 0, :, :]
        # S = S_save.reshape(S.size())
        # bits += bits_s

        # encode H
        R_lv2_star_arg = H.view(R_lv2_star_arg.size(0), 1, hr_lv2.size(2) * hr_lv2.size(3))
        H = R_lv2_star_arg.cpu().numpy()
        H, bits_h = H_conpress.H_compress(H, h=128, w=128, type='x4', idx=idx)
        R_lv2_star_arg = torch.from_numpy(H).cuda()
        bits += bits_h
        ### transfer
        st_time = time.time()
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(3, 3), padding=1)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(6, 6), padding=2, stride=2)

        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv2_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv2_star_arg)

        T_lv2 = F.fold(T_lv2_unfold, output_size=(hr_lv2.size(2), hr_lv2.size(3)), kernel_size=(3, 3),
                       padding=1) / (3. * 3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(hr_lv2.size(2) * 2, hr_lv2.size(3) * 2),
                       kernel_size=(6, 6),
                           padding=2, stride=2) / (3. * 3.)
        ed_time = time.time()
        sr_time = ed_time - st_time
        return S, R_lv2_star_arg, lr, T_lv2, T_lv1, bits

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
    hr = HR * 2 - 1.

    sr_time = 0
    return hr, bits