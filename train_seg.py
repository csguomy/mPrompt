import sys
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from config import config
import net.networks as networks
from eval.Estimator_single import Estimator
from options.train_options import TrainOptions
from Dataset.DatasetConstructor_seg import TrainDatasetConstructor
from ipdb import launch_ipdb_on_exception
import copy

import robust_loss_pytorch.general
from robust_loss_pytorch import lossfun
from robust_loss_pytorch import AdaptiveLossFunction

# Mainly get settings for specific datasets
setting = config(opt)

log_file = os.path.join(setting.model_save_path, opt.dataset_name+'.log')
log_f = open(log_file, "w")

# Data loaders
train_dataset = TrainDatasetConstructor(
    setting.train_num,
    setting.train_img_path,
    setting.train_gt_map_path,
    setting.box_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device,
    is_random_hsi=setting.is_random_hsi,
    is_flip=setting.is_flip,
    fine_size=opt.fine_size,
    opt=opt
    )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=setting.batch_size, shuffle=True, num_workers=opt.nThreads, drop_last=True)

# model construct
net = networks.define_net(opt)
net = net.to(setting.device) # net.cuda()
net = networks.init_net(net, gpu_ids=opt.gpu_ids)

criterion1 = torch.nn.MSELoss(reduction='sum').to(setting.device) # for density map loss
criterion2 = torch.nn.BCELoss().to(setting.device) # for segmentation map loss
estimator = Estimator(opt, setting, eval_loader, criterion=criterion1)

adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction( \
        num_dims = 2 * opt.fine_size * opt.fine_size, \
        float_dtype=np.float32, device=setting.device)

optimizer = networks.select_optim(net, opt, adaptive)
scheduler = networks.get_scheduler(optimizer, opt)


if opt.pretrain:
    print('-----------')
    print('-----------')
    print('-----------')
    print('-----------')
    print('Loading prtrained model:', opt.pretrain_model)
    #if os.path.getsize(opt.pretrain_model) > 0:
    net.module.load_state_dict(torch.load(opt.pretrain_model, map_location=str(setting.device)))

step = 0
base_mae = 10000.0
#----------------------------seg for NWPU-------------------------------
best_loss = 1e6
best_model_wts = copy.deepcopy(net.state_dict())
best_model_name = setting.model_save_path + '/Seg_best_loss_' + str(round(best_loss, 2)) + '_Ep_' + '0' + '.pth'
#----------------------------seg for NWPU-------------------------------

with launch_ipdb_on_exception():

    for epoch_index in range(setting.epoch):
        print('alpha={:03f}  scale={:03f}'.format(adaptive.alpha()[0,0].data, adaptive.scale()[0,0].data)) 
        
        loss_all = 0
        time_per_epoch = 0
        
        for train_img, train_gt, fbs, img_path in train_loader:
            
            # put data to setting.device
            train_img = train_img.to(setting.device)
            train_gt = train_gt.to(setting.device)
            fbs = fbs.to(setting.device)

            net.train()
            x, y = train_img, fbs
            start = time.time()
   
            prediction = net(x)
         
            #loss_seg = criterion2(prediction, fbs)
            x = (prediction - y).view(opt.batch_size, -1)
            loss_seg = torch.mean(adaptive.lossfun(x))            

            optimizer.zero_grad()
            loss_seg.backward()
            loss_seg_item = loss_seg.detach().item()
            optimizer.step()

            step += 1
            end = time.time()
            time_per_epoch += end - start

            if step % opt.print_step == 0:
                print("Step:{:d}\t, Epoch:{:d}\t, Seg Loss:{:.4f}".format(step, epoch_index, loss_seg_item))
            
            loss_all += loss_seg_item
        
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('lr now: %.7f' % lr)
        
        loss_avg = loss_all/train_loader.__len__()

#----------------------------save segmenter for NWPU-------------------------------
        if best_loss > loss_avg:

            best_loss = loss_avg

            best_model_wts = copy.deepcopy(net.state_dict())
            best_model_name = setting.model_save_path + '/Seg_best_loss_' + str(round(best_loss, 2)) + '_Ep_' + str(epoch_index) + '.pth'
        
        print(best_model_name)
        
        if epoch_index % 50 == 0:
            print('Save model:', best_model_name)
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(best_model_wts, best_model_name)
                net.cuda(opt.gpu_ids[0])
            else:
                torch.save(best_model_wts, best_model_name)


    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(best_model_wts, best_model_name)
        net.cuda(opt.gpu_ids[0])
    else:
        torch.save(best_model_wts, best_model_name)
#----------------------------save segmenter for NWPU-------------------------------    

