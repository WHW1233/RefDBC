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

# image_dir = r'G:/WHW/RS_compress/oxford_robotcar/test/patch2/LR_for_com'
image_dir = r'D:\文件DA\科研\视频编码\代码\compression-TTSR\satelite\test\LR_for_com'

image_list = os.listdir(image_dir)

root = 'D:/文件DA/科研/视频编码/代码/bpg'
encoder = 'bpgenc.exe'
decoder = 'bpgdec.exe'

q = 38    # -p 控制压缩的质量     42

_psnr = []
_ssim = []
_bits = []
for img in image_list:
    # if '14_0' not in img:
    #     continue
    print(img)
    img_input = os.path.join(image_dir, img)
    # out_file = os.path.join(r'G:/WHW/RS_compress/oxford_robotcar/test/patch2\bpg_file', img)
    out_file = os.path.join(r'D:\文件DA\科研\视频编码\代码\compression-TTSR\satelite\test\bpg_file', img)
    bpg_file = out_file.replace('.png', '.bpg')
    print(out_file)
    print(bpg_file)
    # compress LR image

    cmd1 = '{}/{} -m 9 -b 8 -q {} {} -o {}'.format(root, encoder, q, img_input, bpg_file)
    t1 = time.time()
    os.system(cmd1)
    t2 = time.time()
    T1 = t2-t1
    cmd2 = '{}/{} -o {} {}'.format(root, decoder, out_file, bpg_file)
    t1 = time.time()
    os.system(cmd2)
    t2 = time.time()
    T2= t2-t1
    with open('G:/WHW/bpg_d_q{}.txt'.format(q), 'a+') as f:
        f.write(str(T2)+"\n")
    HR = imread(img_input)
    HR = HR.astype(np.float32)
    h, w = HR.shape[:2]
    # HR = HR / 127.5 - 1.
    bits = os.path.getsize(bpg_file)
    print(bits)
    bpp = bits * 8 / (h * w)

    SR = imread(out_file)
    SR = SR.astype(np.float32)
    # SR = SR / 127.5 - 1.
    PSNR = calc_psnr(HR, SR)
    SSIM = calc_ssim(HR, SR)
    print(PSNR)
    print(SSIM)
    print(bpp)
    _psnr.append(PSNR)
    _ssim.append(SSIM)
    _bits.append(bpp)
#     # csv_writer.writerow([img, RATE, str(24/RATE), PSNR, SSIM])
#
print('avg psnr:',np.mean(_psnr))
print('avg ssim: ',np.mean(_ssim))
print('avg bpp: ',np.mean(_bits))
# csv_writer.writerow(['Average', RATE, str(24/RATE), np.mean(_psnr), np.mean(_ssim)])