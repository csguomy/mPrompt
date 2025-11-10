import random
import math
import copy
import numpy as np
import sys
from PIL import Image
from metrics import AEBatch, SEBatch
import time
import torch
import scipy.io as scio
import os

class Estimator(object):
    def __init__(self, opt, setting, eval_loader, criterion=torch.nn.MSELoss(reduction="sum")):
        self.datasets_com = ['SHA', 'SHB', 'QNRF', 'NWPU']
        self.setting = setting
        self.ae_batch = AEBatch().to(self.setting.device)
        self.se_batch = SEBatch().to(self.setting.device)
        self.crit_0 = criterion
        self.opt = opt # also load opt
        self.eval_loader = eval_loader
        self.eval_bz_pg = opt.eval_size_per_GPU
        
    def evaluate(self, net_0, epoch_index, total_eval_num): # using name `net_0` for consistency
        
        self.net_0 = net_0.module.eval() # mention: self.net_0 contain DataParrel container
        #self.net_0 = net_0.eval() # mention: self.net_0 do not contain DataParrel container
        
        self.crit_0 = self.crit_0.to(self.opt.gpu_ids[0])
        gpus = len(self.opt.gpu_ids)
        

        net_lst = [self.net_0]
        for ig in range(1, gpus):
            net_lst.append(eval("copy.deepcopy(self.net_0).to(self.opt.gpu_ids[ig])"))

        crit_lst = [self.crit_0]
        for ig in range(1, gpus):
            crit_lst.append(eval("copy.deepcopy(self.crit_0).to(self.opt.gpu_ids[ig])"))

        # for whole set
        MAE_, MSE_, loss_ = [], [], []

        # for eval each single dataset
        imgs_cnt = 0#[0, 0, 0, 0] # logging for each dataset
        pred_mae = 0#[0, 0, 0, 0]
        pred_mse = 0#[0, 0, 0, 0]


        start = time.time()
        is_save = 0
        rand_number, cur, time_cost = random.randint(0, self.setting.eval_num - 1), 0, 0
        for img_shape, eval_img_path, eval_img, mask, eval_gt, pH, pW, gt_h, gt_w, mH, mW in self.eval_loader: 
        
        # for test val of nwpu
        #for eval_img_path, eval_img, pH, pW, eval_gt in self.eval_loader:      
            bz = eval_img.size(0)
            cur += bz
            print('evaluating img_idex: %d / %d in epoch: %d' %(cur, total_eval_num, epoch_index))

            eval_patchs_lst = []
            #eval_patchs_mask_lst = []
            eval_gt_lst = []
            # remove extra dim
            for i in range(bz):
                # for now, eval_patchs_x: bz_little*9*3*H_max*W_max
                # crop the input patchs
                eval_patchs_lst.append(eval("eval_img[i*self.eval_bz_pg: (i+1)*self.eval_bz_pg, :, :, : pH[i], : pW[i]].to(self.opt.gpu_ids[i]).squeeze(0)"))
                # eval_patchs_lst.append(eval("eval_img[i*self.eval_bz_pg: (i+1)*self.eval_bz_pg, :, : pH[i], : pW[i]].to(self.opt.gpu_ids[i])"))

                # crop the mask patchs
                #eval_patchs_mask_lst.append(eval("mask[i*self.eval_bz_pg: (i+1)*self.eval_bz_pg, :, :, : mH[i], : mW[i]].to(self.opt.gpu_ids[i]).squeeze(0)"))
                # for gt_maps, NO NEED TO CROP it, it is fine.
                
                eval_gt_lst.append(eval("eval_gt[i*self.eval_bz_pg: (i+1)*self.eval_bz_pg].to(self.opt.gpu_ids[i]).squeeze(0)"))
                #eval_gt_lst.append(eval("eval_gt[i*self.eval_bz_pg: (i+1)*self.eval_bz_pg].to(self.opt.gpu_ids[i])"))
            prediction_map_lst = [torch.zeros_like(p) for p in eval_gt_lst]

  
            with torch.no_grad():
                eval_prediction_lst = []
                eval_patchs_shape_lst = []

                for i in range(bz):
                    tmp, _= eval("net_lst[i](eval_patchs_lst[i])")
                    # seg mask
                    #eval_patchs_mask_lst[i] = (eval_patchs_mask_lst[i] > 0)
                    #eval_prediction_lst.append(tmp.masked_fill_(~eval_patchs_mask_lst[i], 0))
                    eval_prediction_lst.append(tmp)
                    eval_patchs_shape_lst.append(tmp.shape)

                # test cropped patches
                gt_counts_lst = []
                batch_ae_lst = []
                batch_se_lst = []
                for i in range(bz):
                    self.test_crops(eval_patchs_shape_lst[i], eval_prediction_lst[i], prediction_map_lst[i])
                    #prediction_map_lst[i] = eval_prediction_lst[i]
                    
                    
                    gt_counts_lst.append(self.get_gt_num(eval_img_path[i]))

                    batch_ae_lst.append(self.ae_batch(prediction_map_lst[i], gt_counts_lst[i]).data.cpu().numpy())
                    #batch_ae_lst.append(torch.abs(torch.sum(prediction_map_lst[i], dim=(1, 2, 3)) -gt_counts_lst[i]).data.cpu().numpy())
                    batch_se_lst.append(self.se_batch(prediction_map_lst[i], gt_counts_lst[i]).data.cpu().numpy())
                    #batch_se_lst.append(torch.pow(torch.sum(prediction_map_lst[i]) -gt_counts_lst[i], 2).data.cpu().numpy())
                  
                    MAE_.append(batch_ae_lst[i])
                    MSE_.append(batch_se_lst[i])

                    imgs_cnt += 1
                    pred_mae  += batch_ae_lst[i][0]
                    pred_mse  += batch_se_lst[i][0] # [0] is to convert the array to number~
        

        # synchronize here instead of in the for loop
        torch.cuda.synchronize()
        end = time.time()
        time_cost += (end - start)
        
        pred_mae = pred_mae / imgs_cnt 
        pred_mse = pred_mse / imgs_cnt 
        pred_mse = np.sqrt(pred_mse)

        # return the validate loss, validate MAE and validate RMSE
        MAE_, MSE_= np.reshape(MAE_, [-1]), np.reshape(MSE_, [-1])

        print("time cost: %f" % time_cost)
        return np.mean(MAE_), np.sqrt(np.mean(MSE_)), 0, time_cost, pred_mae, pred_mse
  

    def get_cur_dataset(self, img_name):
        check_list = [img_name.find(da) for da in self.datasets_com]
        check_list = np.array(check_list)
        cur_idx = np.where(check_list != -1)[0][0]
        return self.datasets_com[cur_idx]

    # New Function
    def get_gt_num(self, eval_img_path):
        mat_name = eval_img_path.replace('images', 'ground_truth')[:-4] + ".mat"
        gt_counts = len(scio.loadmat(mat_name)['annPoints'])

        return gt_counts

    # infer the gt mat names from img names
    # Very specific for this repo
    # SHA/SHB: SHA_IMG_85.jpg --> SHA_GT_IMG_85.mat
    # QNRF: QNRF_img_0001.jpg --> QNRF_img_0001_ann.mat
    def get_gt_num_old(self, eval_img_path):
        mat_name = eval_img_path.replace('.jpg', '.mat').replace('images', 'ground_truth')
        cur_dataset = self.get_cur_dataset(mat_name)

        if cur_dataset == "QNRF":
            mat_name = mat_name.replace('.mat', '_ann.mat')
            gt_counts = len(scio.loadmat(mat_name)['annPoints'])
        elif cur_dataset == "SHA" or cur_dataset == "SHB":
            mat_name = mat_name.replace('IMG', 'GT_IMG')
            gt_counts = len(scio.loadmat(mat_name)['image_info'][0][0][0][0][0])
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        return gt_counts

    # For JSTL, this function is not supported
    def show_sample(self, index, gt_counts, pred_counts, eval_gt_map, eval_pred_map):
        if self.setting.dataset_name == "QNRF":
            origin_image = Image.open(self.setting.eval_img_path + "/img_" + ("%04d" % index) + ".jpg")
        elif self.setting.dataset_name == "SHA" or self.setting.dataset_name == "SHB":
            origin_image = Image.open(self.setting.eval_img_path + "/IMG_" + str(index) + ".jpg")
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        show(origin_image, eval_gt_map, eval_pred_map, index)
        sys.stdout.write('The gt counts of the above sample:{}, and the pred counts:{}\n'.format(gt_counts, pred_counts))

    def test_crops(self, eval_shape, eval_p, pred_m):
        
        #print(eval_shape, eval_p.shape, pred_m.shape)
        
        for i in range(3):
            for j in range(3):
                start_h, start_w = math.floor(eval_shape[2] / 4), math.floor(eval_shape[3] / 4)
                valid_h, valid_w = eval_shape[2] // 2, eval_shape[3] // 2
                pred_h = math.floor(3 * eval_shape[2] / 4) + (eval_shape[2] // 2) * (i - 1)
                pred_w = math.floor(3 * eval_shape[3] / 4) + (eval_shape[3] // 2) * (j - 1)
                if i == 0:
                    valid_h = math.floor(3 * eval_shape[2] / 4)
                    start_h = 0
                    pred_h = 0
                elif i == 2:
                    valid_h = math.ceil(3 * eval_shape[2] / 4)

                if j == 0:
                    valid_w = math.floor(3 * eval_shape[3] / 4)
                    start_w = 0
                    pred_w = 0
                elif j == 2:
                    valid_w = math.ceil(3 * eval_shape[3] / 4)
                    
                #print(pred_h, pred_h + valid_h, pred_w, pred_w + valid_w, start_h, start_h + valid_h, start_w, start_w + valid_w)  
                
                pred_m[:, :, pred_h:pred_h + valid_h, pred_w:pred_w + valid_w] += eval_p[i * 3 + j:i * 3 + j + 1, :,start_h:start_h + valid_h, start_w:start_w + valid_w]


