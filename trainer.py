from utils import calc_psnr_and_ssim
from model import Vgg19,MainNet, LTE, SearchTransfer

# import matplotlib.pyplot as plt
import os
import numpy as np
from imageio import imread, imsave
from PIL import Image
import cv2 as cv
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
from dataset import degradation
import math
import csv


class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        # self.ft_net = Fineturn_net(args).to(self.device)      #edit by whw
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(self.vgg19, list(range(self.args.num_gpu)))

        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.MainNet.parameters() if 
             args.num_gpu==1 else self.model.module.MainNet.parameters()),
             "lr": args.lr_rate
            },
            {"params": filter(lambda p: p.requires_grad, self.model.LTE.parameters() if 
             args.num_gpu==1 else self.model.module.LTE.parameters()), 
             "lr": args.lr_rate_lte
            }
        ]
        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            #model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, is_init=False, current_loss=10000):
        self.model.train()
        if (not is_init):
            self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))
        total_rec_loss = 0
        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched)
            LR = sample_batched['LR']
            LR_bic = sample_batched['LR_bic']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref = sample_batched['Ref']
            ref_sr = sample_batched['Ref_sr']
            #sr, S, T_lv3, T_lv2, T_lv1 = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
            sr, S, lr, T_lv2, T_lv1, Point, command, mse_loss, bpp = self.model(LR=LR_bic, lrsr=lr_sr, ref=ref, refsr=ref_sr, hr=hr, idx=i_batch)

            ### calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0) ### flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            if i_batch<20:
                lamda = 10
            else:
                lamda = 1
            lrrec_loss = lamda * self.loss_all['rec_loss'](lr, LR_bic)

            loss = rec_loss + lrrec_loss + bpp

            total_rec_loss += rec_loss.item()
            if (is_print):
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.5f, mes_loss: %.5f, bpp: %.5f,' %(rec_loss.item(), lrrec_loss.item()/lamda, bpp.item()))

            if (not is_init):
                if ('per_loss' in self.loss_all):
                    sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                    with torch.no_grad():
                        hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                    per_loss = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                    loss += per_loss
                    if (is_print):
                        self.logger.info( 'per_loss: %.10f' %(per_loss.item()) )
                if ('tpl_loss' in self.loss_all):
                    sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr)
                    tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss']( sr_lv2, sr_lv1,
                        S, T_lv2, T_lv1)
                    loss += tpl_loss
                    if (is_print):
                        self.logger.info( 'tpl_loss: %.10f' %(tpl_loss.item()) )
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * self.loss_all['adv_loss'](sr, hr)
                    loss += adv_loss
                    if (is_print):
                        self.logger.info( 'adv_loss: %.10f' %(adv_loss.item()) )

            loss.backward()
            self.optimizer.step()

        if ((total_rec_loss <= current_loss) or (current_epoch %5 ==0)):
            current_loss = total_rec_loss if total_rec_loss <= current_loss else current_loss
            print("total rec loss is :", total_rec_loss)
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)
        return current_loss

    def evaluate(self, current_epoch=0, use_gmm = False):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')
        # f = open('paper_result.csv', 'a+', encoding='utf-8', newline='')
        # csv_writer = csv.writer(f)
        if (self.args.dataset == 'SATELITE' or self.args.dataset == 'ROBOTCAR'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
                sum_p = 0
                sum_bits = 0
                bit_medium = 0
                bit_hard = 0
                sum_S = 0
                for i_batch, sample_batched in enumerate(self.dataloader['test']['1']):
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    lr_sr = sample_batched['LR_sr']
                    lr_bic = sample_batched['LR_bic']
                    ref_sr = sample_batched['Ref_sr']

                    sr, S, lr, T_lv2, T_lv1, Point, command, mse_loss, bits = self.model(LR=lr_bic, lrsr=lr_sr, ref=ref, refsr=ref_sr, hr=hr, idx=i_batch)
                    if Point is not None:
                        sum_p+=len(Point)
                    if (self.args.eval_save_results):
                        sr_save = (sr + 1.) * 127.5
                        lr_save = (ref_sr +1.) * 127.5

                        sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5) + '.png'),
                               sr_save)
                        lr_save = np.transpose(lr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        # imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5) + '_lr.png'),
                        #        lr_save)
                        # if i_batch == 12:
                        #     exit()

                    ### calculate psnr and ssim
                    if command is not 'xx':
                        cnt += 1
                        _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())
                        self.logger.info('batch %f PSNR & SSIM: %.2f, %.4f' %(i_batch, _psnr, _ssim))
                        self.logger.info('mean S is %.4f, and command is %s, bpp is %.4f' % (torch.mean(S), command, bits))
                        # csv_writer.writerow([i_batch, '{:.4f}'.format(bits.detach().cpu().numpy()),'{:.4f}'.format(_psnr),'{:.4f}'.format(_ssim)])
                        psnr += _psnr
                        ssim += _ssim
                        sum_bits += bits
                        if command == '90':
                            bit_medium += bits
                        if command == '70':
                            bit_hard += bits
                        sum_S += torch.mean(S)
                        if use_gmm:
                            '计算bits'
                        # if i_batch in [41, 47, 53, 70, 71, 75, 78]:
                        #     with open('recode.txt', 'a+') as ftxt:
                        #         ftxt.write('%d, psnr: %.3f, S: %.3f, command:%s \n'
                        #             % (i_batch,_psnr,torch.mean(S),command))

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                S_ave = sum_S / cnt
                if isinstance(sum_bits, float):
                    bpp = sum_bits
                else:
                    bpp = sum_bits.detach().cpu().numpy()
                self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f' % (psnr_ave, ssim_ave))
                bpp_ave = bpp / cnt
                self.logger.info('Ref bpp(now): %.4f' % bpp_ave)

                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)'
                                 % (self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))
                # psnr_list.append(str(psnr_ave))
                # psnr_list.append(bpp/cnt)
                # csv_writer.writerow(psnr_list)
        self.logger.info('Evaluation over.')

    def test(self, use_gmm=False):
        self.logger.info('Test process...')
        self.logger.info('lr path:     %s' % (self.args.lr_path))
        self.logger.info('ref path:    %s' % (self.args.ref_path))

        ### data loading
        HR = imread(self.args.lr_path)
        HR = HR[:2160, :4096, :]
        h, w = HR.shape[:2]
        h, w = h // 2 * 2, w // 2 * 2
        HR = HR[:h, :w, :]

        # blur
        kernel_size = 7
        sigma = 1
        kernel = degradation.bivariate_Gaussian(kernel_size, sigma, 5, -math.pi, isotropic=True)
        hr_img = cv.filter2D(HR, -1, kernel=kernel)

        LR = np.array(Image.fromarray(hr_img).resize((w // 2, h // 2), Image.BICUBIC))  # 4倍降采样
        h1, w1 = LR.shape[:2]
        LR_sr = np.array(Image.fromarray(LR).resize((w1 * 2, h1 * 2), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref = imread(self.args.ref_path)
        Ref = Ref[:2160, :4096, :]
        img_save = Ref
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2 // 2 * 2, w2 // 2 * 2
        Ref = img_save[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2 // 4, h2 // 4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

        ### change type
        HR = HR.astype(np.float32)
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        HR = HR / 127.5 - 1.
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        ### to tensor
        t1 = time.time()
        self.model.eval()
        scale = 256
        scale2=scale//2
        num_h = h//scale
        num_w = w//scale
        # exit()
        for i in range(num_h):
            for j in range(num_w):
                LR_patch = LR[i*scale2:(i+1)*scale2,j*scale2:(j+1)*scale2,:]
                LR_sr_patch = LR_sr[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale,:]
                Ref_patch = Ref[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale,:]
                Ref_sr_patch = Ref_sr[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale,:]
                HR_patch = HR[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale,:]

                LR_t = torch.from_numpy(LR_patch.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)
                LR_sr_t = torch.from_numpy(LR_sr_patch.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)
                Ref_t = torch.from_numpy(Ref_patch.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)
                Ref_sr_t = torch.from_numpy(Ref_sr_patch.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)
                HR_t = torch.from_numpy(HR_patch.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    sr, S, lr, T_lv2, T_lv1, Point, command, mse_loss, bits = self.model(LR=LR_t, lrsr=LR_sr_t, ref=Ref_t,
                                                                                         refsr=Ref_sr_t, hr=HR_t, idx=0)
                    sr_save = (sr + 1.) * 127.5
                    sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                    img_save[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale,:] = sr_save
        save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path))
        imsave(save_path, img_save)
        self.logger.info('output path: %s' % (save_path))

        t2 = time.time()
        T = t2-t1
        self.logger.info('time: %f s' % (T))
        self.logger.info('Test over.')
