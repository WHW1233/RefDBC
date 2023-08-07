from option import args
from utils import mkExpDir
from dataset import dataloader
from model import TTSR
from loss.loss import get_loss_dict
from trainer import Trainer

import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ### make save_dir
    _logger = mkExpDir(args)

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None

    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    _model = TTSR.TTSR(args).to(device)
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### loss
    _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.eval):
        t.load(model_path=args.model_path)
        t.evaluate(use_gmm=args.usegmm)
    elif (args.finetune):
        t.load(model_path=args.model_path)
        current_loss = 10000      # edit bu whw 8/4 2021
        for epoch in range(1, args.num_init_epochs + 1):
            current_loss = t.train(current_epoch=epoch, is_init=True, current_loss=current_loss)
        for epoch in range(1, args.num_epochs + 1):
            current_loss = t.train(current_epoch=epoch, is_init=False, current_loss=current_loss)
            # if (epoch % args.val_every == 0):
            #     t.evaluate(current_epoch=epoch)
    else:
        current_loss = 10000
        for epoch in range(1, args.num_init_epochs+1):
            current_loss = t.train(current_epoch=epoch, is_init=True, current_loss=current_loss)
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False, current_loss=current_loss)
            # if (epoch % args.val_every == 0):
            #     t.evaluate(current_epoch=epoch)
