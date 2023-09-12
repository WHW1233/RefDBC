import os
from model.GMM_com.model_GMM import *
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import json
from tensorboardX import SummaryWriter
# from Meter import AverageMeter
import pdb
import cv2 as cv
from imageio import imsave

torch.backends.cudnn.enabled = True
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 8192
print_freq = 100
cal_step = 40
warmup_step = 0#  // gpu_num
batch_size = 1
tot_epoch = 1000000
tot_step = 2500000
decay_interval = 2200000
lr_decay = 0.1
image_size = 256
# logger = logging.getLogger("ImageCompression")
tb_logger = None
global_step = 0
save_model_freq = 50000
test_step = 10000
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




def test_net(net, input, idx=0):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        name = 'test_image_{}.png'.format(idx)

        clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(input)
        bits = bpp*input.size(-2)*input.size(-1)/8        # 计算出来降采样后的大小
        LR = clipped_recon_image*255/127.5-1
        save_image = clipped_recon_image * 255

        sr_save = np.transpose(save_image.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        imsave(r'G:\WHW\论文投稿\2023-tgrs\23年7月修改\数据集修改\GMM压缩_1024.png',sr_save)
        mse_loss, bpp_feature, bpp_z, bpp = \
            torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
        psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
        sumBpp += bpp
        sumPsnr += psnr
        cnt += 1

        sumBpp /= cnt
        sumPsnr /= cnt
        print("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}".format(sumBpp, sumPsnr))
        return LR, bits, bpp, psnr


# 需要参数 seed不用，name -n，config，pretrain，val测试数据集路径
def use_GMM_com(LR, pretrain_path='./pretrained_model/', com_level=0, idx=0):
    # 默认值0表示4096_256
    L = Com_level(com_level, pretrain_path)
    # pretrain_path = L.get_model_path()
    print("image compression val")
    out_channel_N = L.get_channel()
    out_channel_M = L.get_channel()
    model = ImageCompressor(out_channel_N)
    # pdb.set_trace()
    if pretrain_path != '':
        print("loading model:{}".format(pretrain_path))
        global_step = load_model(model, pretrain_path)
    net = model.cuda()
    # net = torch.nn.DataParallel(net, list(range(gpu_num)))

    # flops, params = profile(net, inputs=(LR,))
    # print(f"GMM 的 FLOPs: {flops}")

    clipped_recon_image, bits, bpp, psnr = test_net(net, LR, idx=idx)
    return clipped_recon_image, bits, bpp, psnr

