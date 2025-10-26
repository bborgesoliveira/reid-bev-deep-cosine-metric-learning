"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

import tools.lss_transform.src as src
from src.models import compile_model
from src.data import compile_data
from src.tools import SimpleLoss, get_batch_iou, get_val_info
import torch.nn.functional as F


def train(version,
            dataroot='/data/nuscenes',
            nepochs=10000,
            gpuid=1,

            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            ncams=6,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=4,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7,
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    #Altera outC para 3, para permitir a saída de 3 canais de características no final.
    model = compile_model(grid_conf, data_aug_conf, outC=64)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 1000 if version == 'mini' else 10000
    
    # Definindo a camada convolucional que reduz os canais de 64 para 1
    conv_layer = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
    conv_layer.to(device)

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            x, preds_multi_channel, preds_one_channel = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )

            binimgs = binimgs.to(device)
            imgs = imgs.to(device)
            # print(imgs.shape)
            # print(binimgs.shape)
            # print(preds_three_channel.shape)
            # print(x.shape)
            # with open('tensor.npy', 'wb') as f:
            #     np.save(f, x[0].cpu().detach().numpy())
            # with open('tensor_out_three.npy', 'wb') as f:
            #     np.save(f, preds_three_channel[0].cpu().detach().numpy())
            # with open('binimg.npy', 'wb') as f:
            #     np.save(f, binimgs[0].cpu().detach().numpy())
            #exit()

            #Aplica camada de convolução para transformar a imagem de 3 canais para 1, 
            #a fim de efetuar a comparação com a imagem binária (representando fundo vs objeto)
            #durante o treinamento.
            #preds = conv_layer(preds)

            loss_binarity = loss_fn(preds_one_channel, binimgs) 
            loss_preservation = F.mse_loss(preds_multi_channel, x)
            total_loss = 0.5*loss_binarity + 0.5*loss_preservation
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, total_loss.item())
                writer.add_scalar('train/loss', total_loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds_one_channel, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
                
def val(version,
            dataroot='/data/nuscenes',
            nvals=10,   #number of validations to execute
            gpuid=1,

            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            ncams=6,
            max_grad_norm=5.0,
            pos_weight=2.13,

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=4,
            nworkers=10,
            modelf='/lasi/cosine_metric_learning/runs/dual_loss_64channels_all/model290000.pt'
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    #Altera outC para 3, para permitir a saída de 3 canais de características no final.
    model = compile_model(grid_conf, data_aug_conf, outC=64)
    #modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/dual_loss_64channel/model830000.pt'
    #modelf = '/lasi/cosine_metric_learning/runs/dual_loss_64channels_vehicles/model290000.pt'
    #modelf = '/lasi/cosine_metric_learning/runs/dual_loss_64channels_all/model290000.pt'
    model.load_state_dict(torch.load(modelf))
    model.eval()
    model.to(device)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    iou_list = np.array([])
    for epoch in range(nvals):            
        t1 = time()

        val_info = get_val_info(model, valloader, loss_fn, device) 
        iou_list = np.append(iou_list, float(val_info['iou']))
        print(iou_list)
        print('VAL', val_info)
        print(f'iou: min = {np.min(iou_list)}\tmax = {np.max(iou_list)}\tmean = {np.average(iou_list)}')