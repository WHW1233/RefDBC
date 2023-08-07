import os
from utils import calc_psnr, calc_ssim
from imageio import imread, imsave
import numpy as np
import csv
from dataset import degradation
import cv2 as cv
from PIL import Image
import time

def get_lrimg(img, d_scale=2):
    HR = imread(img)
    h, w = HR.shape[:2]

    # blur
    kernel_size = 7
    sigma = 1
    kernel = degradation.bivariate_Gaussian(kernel_size, sigma, 5, -3.14, isotropic=True)
    hr_img = cv.filter2D(HR, -1, kernel=kernel)
    # hr_img = HR

    ### LR and LR_sr
    # hr_img = cv.GaussianBlur(HR, (0, 0), 1, 1)
    LR_bic = np.array(Image.fromarray(hr_img).resize((w // d_scale, h // d_scale), Image.BICUBIC))
    return LR_bic



# fcsv = open('JPEG2000_satellite.csv', 'a+', encoding='utf-8', newline='')
# csv_writer = csv.writer(fcsv)
# csv_writer.writerow(["image", "quality", "bpp", 'psnr'])

input_file_name = 'LR_for_com'

# image_dir = 'G:/WHW/RS_compress/oxford_robotcar/test/LR_for_com'
# J2K_dir = 'G:/WHW/RS_compress/oxford_robotcar/test/J2K_file'
# out_dir = 'G:/WHW/RS_compress/oxford_robotcar/test/rec_img'

image_dir = 'D:/文件DA/科研/视频编码/代码/compression-TTSR/satelite/test/LR_for_com'
J2K_dir = 'D:/文件DA/科研/视频编码/代码/compression-TTSR/satelite/test/J2K_file'
out_dir = 'D:/文件DA/科研/视频编码/代码/compression-TTSR/satelite/test/rec_img'
image_list = os.listdir(image_dir)

root = 'D:/文件DA/科研/视频编码/代码/jpeglib/openjpeg-v2.3.0-windows-x64/openjpeg-v2.3.0-windows-x64/bin'
# cmd1 = '{}/opj_compress.exe -i ./compression_data/S/S_{}.png \
#             -o ./compression_data/encode/lr_S_{}.J2K -r {}'.format(root, idx, idx, S_rate)

P = 30      # -p 控制压缩的PSNR
RATE = 12 # 96   # -r 控制压缩倍数
_psnr = []
_ssim = []
_bits = []
for img in image_list:
    print(img)

    img_input = os.path.join(image_dir, img)
    img_output = img_input.replace(input_file_name, 'J2K_file')
    img_output = img_output.replace('.png', '.J2K')
    print(img_output)
    # compress LR image
    # LR = get_lrimg(img_input)
    # imsave('temp.png', LR)
    # img_input = 'temp.png'
    t1 = time.time()
    cmd1 = '{}/opj_compress.exe -i {} \
                -o {} -r {}'.format(root, img_input, img_output, RATE)
    os.system(cmd1)
    t2 = time.time()
    with open('G:/WHW/j2_e_{}.txt'.format(RATE), 'a+') as f:
        f.write(str(t2-t1)+'\n')
    bits = os.path.getsize(img_output)
    bpp = bits * 8 / (256 * 256)
    decode_img = img_output.replace('J2K_file', 'rec_img')
    decode_img = decode_img.replace('.J2K', '.png')
    cmd2 = '{}/opj_decompress.exe -i {} -o {}'.format(root, img_output, decode_img)
    t1 = time.time()
    os.system(cmd2)
    t2 = time.time()
    with open('G:/WHW/j2_d_{}.txt'.format(RATE), 'a+') as f:
        f.write(str(t2-t1)+'\n')
    HR = imread(img_input)
    HR = HR.astype(np.float32)
    # HR = HR / 127.5 - 1.

    SR = imread(decode_img)
    SR = SR.astype(np.float32)
    # os.remove('temp.png')
    # SR = SR / 127.5 - 1.
    # PSNR = calc_psnr(HR, SR)
    SSIM = calc_ssim(HR, SR)
    # print(PSNR)
    print(SSIM)
    print(bpp)
    # _psnr.append(PSNR)
    _ssim.append(SSIM)
    _bits.append(bpp)
    # csv_writer.writerow([img, RATE, str(24/RATE), PSNR, SSIM])

# print('avg PSNR',np.mean(_psnr))
# print('avg SSIM',np.mean(_ssim))
print('avg bpp',np.mean(_bits))
# csv_writer.writerow(['Average', RATE, str(24/RATE), np.mean(_psnr), np.mean(_ssim)])