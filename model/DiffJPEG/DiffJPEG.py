# Pytorch
import torch
import torch.nn as nn
# Local
from model.DiffJPEG.modules import compress_jpeg, decompress_jpeg
from model.DiffJPEG.utils2 import diff_round, quality_to_factor
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import calc_psnr

class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        '''

        '''
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered

if __name__ == "__main__":
    LR_path = 'D:/文件DA/科研/视频编码/代码/TTSR_and_GMM_model/satellite_patch/train/input/Valencia_port_raw_142_time_20110520.png'
    LR = Image.open(LR_path).convert('RGB')
    # LR.show()
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    LR = transform(LR).cuda()
    print(LR.size())
    h,w = LR.shape[-2:]
    print(h,w)
    jpeg = DiffJPEG(h, w, differentiable=True, quality=70)
    jpeg = jpeg.cuda()
    LR = LR.reshape((1, LR.size(0), LR.size(1), LR.size(2)))
    print(LR.size())
    out_img = jpeg(LR)

    out_img = out_img.reshape((3,h,w))
    out_img *= 255
    print(out_img.shape)
    Img = np.transpose(out_img.squeeze().round().cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8)
    print(Img.shape)
    Img = Image.fromarray(Img).convert('RGB')
    Img.save('test2.png')
    LR = LR.reshape((3,h,w))*255
    img1 = np.transpose(LR.squeeze().round().cpu().detach().numpy(), (1,2,0))
    img2 = np.transpose(out_img.squeeze().round().cpu().detach().numpy(), (1, 2, 0))
    _psnr = calc_psnr(img1, img2)

    print(_psnr)




