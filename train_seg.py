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
from Dataset.DatasetConstructor_seg import TrainDatasetConstructor,EvalDatasetConstructor
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
eval_dataset = EvalDatasetConstructor(
    setting.eval_num,
    setting.eval_img_path,
    setting.eval_gt_map_path,
    setting.box_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=setting.batch_size, shuffle=True, num_workers=opt.nThreads, drop_last=True)

def my_collfn(batch):
    img_path = [item[0] for item in batch]
    imgs = [item[1] for item in batch]
    gt_map = [item[2] for item in batch]
    gt_H = [item[3] for item in batch]
    gt_W = [item[4] for item in batch]
    pH = [item[5] for item in batch]
    pW = [item[6] for item in batch]

    bz = len(batch)

    gt_H = torch.stack(gt_H, 0)
    gt_W = torch.stack(gt_W, 0)
    pH = torch.stack(pH, 0)
    pW = torch.stack(pW, 0)
    gt_h_max = torch.max(gt_H)
    gt_w_max = torch.max(gt_W)

    ph_max = torch.max(pH)
    pw_max = torch.max(pW)

    imgs_new = torch.zeros(bz, 9, 3, ph_max, pw_max) # bz * 9 * c * gth_max * gtw_max
    gt_map_new = torch.zeros(bz, 1, 1, gt_h_max, gt_w_max)

    # put map
    for i in range(bz):
        imgs_new[i, :, :, :pH[i], :pW[i]] = imgs[i]
        # h, w
        gt_map_new[i, :, :, :gt_H[i], :gt_W[i]] = gt_map[i]

    return img_path, imgs_new, gt_map_new, pH, pW, gt_H, gt_W

assert opt.eval_size_per_GPU == 1, "Using this is fast enough and for large size evaluation"
batch_eval_size = opt.eval_size_per_GPU * len(opt.gpu_ids)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_eval_size, collate_fn=my_collfn)

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
        
        loss_all = 0
        print('alpha={:03f}  scale={:03f}'.format(adaptive.alpha()[0,0].data, adaptive.scale()[0,0].data)) 
        
        # eval
        if epoch_index % opt.eval_per_epoch == 0 and epoch_index > opt.start_eval_epoch:
            
            print('Evaluating epoch:', str(epoch_index))
            torch.cuda.empty_cache()
           
            validate_MAE, validate_RMSE, validate_loss, time_cost, pred_mae, pred_mse = estimator.evaluate(net,  epoch_index, eval_loader.__len__() * batch_eval_size) 
            # validate the code of eval
            if opt.test_eval==1 and opt.start_eval_epoch==-1:
                assert 1==2, print('Test over~')
            
            log_f.write(
                'In step {}, epoch {}, loss = {}, eval_mae = {}, eval_rmse = {}, mae = {}, mse = {}, time cost eval = {}s\n'.format(step, epoch_index, validate_loss, validate_MAE, validate_RMSE, pred_mae,
                        pred_mse, time_cost))
            log_f.flush()

            # save model with epoch and MAE
            save_now = False
            if pred_mae < base_mae:
                save_now = True
            if save_now:
                best_model_name = setting.model_save_path + "/MAE_" + str(round(validate_MAE, 2)) + \
                    "_MSE_" + str(round(validate_RMSE, 2)) + '_mae_' + str(round(pred_mae, 2)) + \
                    '_mse_' + str(round(pred_mse, 2)) + \
                    '_Ep_' + str(epoch_index) + '.pth'
                if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), best_model_name)
                    net.cuda(opt.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), best_model_name)


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

