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
        


    def evaluate(self, net_0, is_show=True): # using name `net_0` for consistency
        net_0.eval()
        gpus = len(self.opt.gpu_ids)
        for ig in range(gpus-1):
            cur_net_name = "net_"+repr(ig+1)
            cur_crit_name = "crit_"+repr(ig+1)
            setattr(self, cur_net_name, eval("copy.deepcopy(net_0).module.to(self.opt.gpu_ids[ig])"))
            setattr(self, cur_crit_name, eval("copy.deepcopy(self.crit_0).to(self.opt.gpu_ids[ig])"))

        MAE_, MSE_, loss_ = [], [], []

        # for eval each single dataset
        imgs_cnt = [0, 0, 0, 0] # logging for each dataset
        pred_mae = [0, 0, 0, 0]
        pred_mse = [0, 0, 0, 0]

        rand_number, cur, time_cost = random.randint(0, self.setting.eval_num - 1), 0, 0
        for eval_img_path, eval_img, eval_gt, class_id in self.eval_loader:
            bz = eval_img.size(0)
            assert bz == 4, "evaluation batch size must be 4"
            class_id = class_id.to(self.setting.device)

            start = time.time()
            # remove extra dim
            for i in range(bz):
                cur_eval_patchs = "eval_patchs_" + repr(i)
                cur_eval_gt = "eval_gt_" + repr(i)
                # for now, eval_patchs_x: bz_little*9*3*H_max*W_max
                setattr(self, cur_eval_patchs, eval("eval_img[i*self.eval_bz_pg: (i+1)*self.eval_bz_pg].to(self.opt.gpu_ids[i])"))
                setattr(self, cur_eval_gt, eval("eval_gt[i*self.eval_bz_pg: (i+1)*self.eval_bz_pg].to(self.opt.gpu_ids[i])"))

            # ugly code, but easy to write...
            # TODO: rewrite it more elegant
            eval_gt_shape_0 = eval_gt_0.shape
            eval_gt_shape_1 = eval_gt_1.shape
            eval_gt_shape_2 = eval_gt_2.shape
            eval_gt_shape_3 = eval_gt_3.shape
            
            prediction_map_0 = torch.zeros_like(eval_gt_0)
            prediction_map_1 = torch.zeros_like(eval_gt_1)
            prediction_map_2 = torch.zeros_like(eval_gt_2)
            prediction_map_3 = torch.zeros_like(eval_gt_3)
            
            eval_img_path_0 = eval_img_path[0]
            eval_img_path_1 = eval_img_path[1]
            eval_img_path_2 = eval_img_path[2]
            eval_img_path_3 = eval_img_path[3]
            

            with torch.no_grad():
                eval_prediction_0, _ = net_0(eval_patchs_0)
                eval_prediction_1, _ = net_1(eval_patchs_1)
                eval_prediction_2, _ = net_2(eval_patchs_2)
                eval_prediction_3, _ = net_3(eval_patchs_3)

                eval_patchs_shape_0 = eval_prediction_0.shape
                eval_patchs_shape_1 = eval_prediction_1.shape
                eval_patchs_shape_2 = eval_prediction_2.shape
                eval_patchs_shape_3 = eval_prediction_3.shape
                
                # test cropped patches
                self.test_crops(eval_patchs_shape_0, eval_prediction_0, prediction_map_0)
                self.test_crops(eval_patchs_shape_1, eval_prediction_1, prediction_map_1)
                self.test_crops(eval_patchs_shape_2, eval_prediction_2, prediction_map_2)
                self.test_crops(eval_patchs_shape_3, eval_prediction_3, prediction_map_3)
                
                gt_counts_0 = self.get_gt_num(eval_img_path_0)
                gt_counts_1 = self.get_gt_num(eval_img_path_1)
                gt_counts_2 = self.get_gt_num(eval_img_path_2)
                gt_counts_3 = self.get_gt_num(eval_img_path_3)
                
                # calculate metrics
                batch_ae_0 = self.ae_batch(prediction_map_0, gt_counts_0).data.cpu().numpy()
                batch_ae_1 = self.ae_batch(prediction_map_1, gt_counts_1).data.cpu().numpy()
                batch_ae_2 = self.ae_batch(prediction_map_2, gt_counts_2).data.cpu().numpy()
                batch_ae_3 = self.ae_batch(prediction_map_3, gt_counts_3).data.cpu().numpy()
                
                batch_se_0 = self.se_batch(prediction_map_0, gt_counts_0).data.cpu().numpy()
                batch_se_1 = self.se_batch(prediction_map_1, gt_counts_1).data.cpu().numpy()
                batch_se_2 = self.se_batch(prediction_map_2, gt_counts_2).data.cpu().numpy()
                batch_se_3 = self.se_batch(prediction_map_3, gt_counts_3).data.cpu().numpy()

                loss_0 = self.criterion(prediction_map_0, eval_gt_0)
                loss_1 = self.criterion(prediction_map_1, eval_gt_1)
                loss_2 = self.criterion(prediction_map_2, eval_gt_2)
                loss_3 = self.criterion(prediction_map_3, eval_gt_3)
                
                loss_.append(loss_0.data.item())
                loss_.append(loss_1.data.item())
                loss_.append(loss_2.data.item())
                loss_.append(loss_3.data.item())
                
                MAE_.append(batch_ae_0)
                MAE_.append(batch_ae_1)
                MAE_.append(batch_ae_2)
                MAE_.append(batch_ae_3)
                
                MSE_.append(batch_se_0)
                MSE_.append(batch_se_1)
                MSE_.append(batch_se_2)
                MSE_.append(batch_se_3)

                # bz=4
                imgs_cnt[class_id[0].item()] += 1
                pred_mae[class_id[0].item()] += batch_ae_0[0]
                pred_mse[class_id[0].item()] += batch_ae_0[0]

                imgs_cnt[class_id[1].item()] += 1
                pred_mae[class_id[1].item()] += batch_ae_1[0]
                pred_mse[class_id[1].item()] += batch_ae_1[0]
                
                imgs_cnt[class_id[2].item()] += 1
                pred_mae[class_id[2].item()] += batch_ae_2[0]
                pred_mse[class_id[2].item()] += batch_ae_2[0]
                
                imgs_cnt[class_id[3].item()] += 1
                pred_mae[class_id[3].item()] += batch_ae_3[0]
                pred_mse[class_id[3].item()] += batch_ae_3[0]

            torch.cuda.synchronize()
            end = time.time()
            time_cost += (end - start)

        # cal mae, mse for each dataset
        pred_mae = np.array(pred_mae)
        pred_mse = np.array(pred_mse)
        imgs_cnt = np.array(imgs_cnt)

        pred_mae = pred_mae / imgs_cnt
        pred_mse = pred_mse / imgs_cnt
        pred_mse = np.sqrt(pred_mse)

        # return the validate loss, validate MAE and validate RMSE
        MAE_, MSE_, loss_ = np.reshape(MAE_, [-1]), np.reshape(MSE_, [-1]), np.reshape(loss_, [-1])
        return np.mean(MAE_), np.sqrt(np.mean(MSE_)), np.mean(loss_), time_cost, pred_mae, pred_mse

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
                pred_m[:, :, pred_h:pred_h + valid_h, pred_w:pred_w + valid_w] += eval_p[i * 3 + j:i * 3 + j + 1, :,start_h:start_h + valid_h, start_w:start_w + valid_w]
