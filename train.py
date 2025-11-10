import sys
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
import net.networks as networks
from eval.Estimator import Estimator
from options.train_options import TrainOptions
from Dataset.DatasetConstructor import TrainDatasetConstructor,EvalDatasetConstructor
from ipdb import launch_ipdb_on_exception
import copy

opt = TrainOptions().parse()
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
    img_shape = [item[0] for item in batch]
    img_path = [item[1] for item in batch]
    imgs = [item[2] for item in batch]
    masks = [item[3] for item in batch]
    gt_map = [item[4] for item in batch]
    gt_H = [item[5] for item in batch]
    gt_W = [item[6] for item in batch]
    pH = [item[7] for item in batch]
    pW = [item[8] for item in batch]
    mH = [item[9] for item in batch]
    mW = [item[10] for item in batch]

    gt_H = torch.stack(gt_H, 0)
    gt_W = torch.stack(gt_W, 0)
    pH = torch.stack(pH, 0)
    pW = torch.stack(pW, 0)
    mH = torch.stack(mH, 0)
    mW = torch.stack(mW, 0)

    gt_h_max = torch.max(gt_H)
    gt_w_max = torch.max(gt_W)
    ph_max = torch.max(pH)
    pw_max = torch.max(pW)
    mh_max = torch.max(mH)
    mw_max = torch.max(mW)
    
    bz = len(batch)
    imgs_new = torch.zeros(bz, 9, 3, ph_max, pw_max) # bz * 9 * c * gth_max * gtw_max
    mask_new = torch.zeros(bz, 9, 1, mh_max, mw_max)
    gt_map_new = torch.zeros(bz, 1, 1, gt_h_max, gt_w_max)
    
    # put map
    for i in range(bz):
        imgs_new[i, :, :, :pH[i], :pW[i]] = imgs[i]
        mask_new[i, :, :, :mH[i], :mW[i]] = masks[i]
        # h, w
        gt_map_new[i, :, :, :gt_H[i], :gt_W[i]] = gt_map[i]
    return img_shape, img_path, imgs_new, mask_new, gt_map_new, pH, pW, gt_H, gt_W, mH, mW

assert opt.eval_size_per_GPU == 1, "Using this is fast enough and for large size evaluation"
batch_eval_size = opt.eval_size_per_GPU * len(opt.gpu_ids)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_eval_size, collate_fn=my_collfn)

# model construct
net = networks.define_net(opt)
net = net.to(setting.device) # net.cuda()
net = networks.init_net(net, gpu_ids=opt.gpu_ids)

if opt.model_ema:
    from timm.utils import get_state_dict, ModelEma
    print('~~ema~~')
    # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    net_ema = ModelEma(
        net,
        decay=opt.model_ema_decay,
        device='cpu' if opt.model_ema_force_cpu else None,
)

opt.niter = train_dataset.__len__()//opt.batch_size//len(opt.gpu_ids)    
criterion1 = torch.nn.MSELoss(reduction='sum').to(setting.device) # for density map loss
criterion2 = torch.nn.BCELoss().to(setting.device) # for segmentation map loss
criterion3 = nn.L1Loss().to(setting.device) # for count loss

estimator = Estimator(opt, setting, eval_loader, criterion=criterion1)
optimizer = networks.select_optim(net, opt)
scheduler = networks.get_scheduler(optimizer, opt)


if opt.pretrain:
    print('-----------')
    print('-----------')
    print('-----------')
    print('-----------')
    print('Loading prtrained model:', opt.pretrain_model)
    #if os.path.getsize(opt.pretrain_model) > 0:
    net.module.load_state_dict(torch.load(opt.pretrain_model, map_location=str(setting.device)), strict=True)

step = 0
best_mae = 1000
best_mse = 1000
with launch_ipdb_on_exception():

    for epoch_index in range(setting.epoch):
        # evalution
        if epoch_index % opt.eval_per_epoch == 0 and epoch_index >= opt.start_eval_epoch:
            print('Evaluating epoch:', str(epoch_index))
            torch.cuda.empty_cache()

            if opt.model_ema:
                net_eval = net_ema.ema
            else:
                net_eval = net

            validate_MAE, validate_RMSE, validate_loss, time_cost, pred_mae, pred_mse = estimator.evaluate(net_eval,  epoch_index, eval_loader.__len__() * batch_eval_size, opt.net_name) 
            
            if validate_MAE < best_mae:
                best_mae = validate_MAE
                best_mse = validate_RMSE
            
            # validate the code of eval
            if opt.test_eval==1 and opt.start_eval_epoch==-1:
                assert 1==2, print('Test over~')
            
            log_f.write(
                'In step {}, epoch {}, loss = {}, eval_mae = {}, eval_rmse = {}, mae = {}, mse = {}, time cost eval = {}s, best_mae = {}_{}\n'.format(step, epoch_index, validate_loss, validate_MAE, validate_RMSE, pred_mae,
                        pred_mse, time_cost, best_mae, best_mse))
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
                    torch.save(net_eval.module.cpu().state_dict(), best_model_name)
                    net_eval.cuda(opt.gpu_ids[0])
                else:
                    torch.save(net_eval.cpu().state_dict(), best_model_name)


        time_per_epoch = 0
        for train_img, train_gt, mask, mask_knn, img_path in train_loader:
            
            # put data to setting.device
            train_img = train_img.to(setting.device)
            train_gt = train_gt.to(setting.device)
            mask = mask.to(setting.device)
            mask_knn = mask_knn.to(setting.device)

            net.train()
            x, y = train_img, train_gt
            start = time.time()

            # den loss
            #prediction = net(x)
            prediction, mask_out = net(x)
                          
            loss = criterion1(prediction, y)
            
            # den_mask
            prediction_mask = (prediction > 0.0).float()
            # OR 
            #prediction_mask = (prediction > prediction.mean()).float()
            
            # iou loss            
            mask_and = prediction_mask.masked_fill(~mask.bool(), 0)
            #mask_and = ((prediction_mask * mask) > 0.0).float()
            #print(mask_and.float().sum((1,2,3)), mask.float().sum((1,2,3)))
            mask_or = ((prediction_mask + mask) > 0.0).float()
            loss_iou = 1.0 - (mask_and.sum((1,2,3))/(mask_or.sum((1,2,3)) + 0.0001)).mean()            
            
            
            # adaptive refinement
            # --------------------------------
            # add prediction blob to segmask_gt
            if epoch_index >= 50:
                #prediction_mask = (den_mask > 0.0).float()#.cpu().detach().numpy()
                #mask = torch.from_numpy(np.bitwise_or(mask, prediction_mask)).float().to(setting.device)
                mask = ((mask + prediction_mask) > 0.0).float()
                mask = mask.masked_fill(~mask_knn, 0)
            # --------------------------------  
            
            # seg loss after adaptive refinement
            loss_seg = criterion2(mask_out, mask)
        
            optimizer.zero_grad()
            (loss + count_loss + loss_seg + loss_iou).backward()

            # update ema
            if opt.model_ema:
                #print('~~ema update~~')
                net_ema.update(net)

            loss_item = loss.detach().item()
            loss_seg_item = loss_seg.detach().item()
            loss_iou_item = loss_iou.detach().item()
            loss_count_item = count_loss.detach().item()
            optimizer.step()

            step += 1
            end = time.time()
            time_per_epoch += end - start

            if step % opt.print_step == 0:
                print("Step:{:d}\t, Epoch:{:d}\t, Loss:{:.4f}, Seg Loss:{:.4f}, IOU Loss:{:.4f}, Count Loss:{:.4f}".format(step, epoch_index, loss_item, loss_seg_item, loss_iou_item, loss_count_item))  
                
        scheduler.step()
