# config
import torch
from config import config
from net.networks import *
from options.test_options import TestOptions
#import torch.utils.data as data
from Dataset.DatasetConstructor import EvalDatasetConstructor
from eval.Estimator_single import Estimator



opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batch_size = 1  # test code only supports batchSize = 1
opt.is_flip = 0  # no flip

# Mainly get settings for specific datasets
setting = config(opt)

# Data loaders
# test
eval_dataset = EvalDatasetConstructor(
    setting.eval_num,
    setting.eval_img_path,
    setting.eval_gt_map_path,
    setting.box_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device)

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

criterion = torch.nn.MSELoss(reduction='sum').to(setting.device) 
estimator = Estimator(opt, setting, eval_loader, criterion=criterion)

#model construct
net = define_net(opt)
net = init_net(net, gpu_ids=opt.gpu_ids)
net.module.load_state_dict(torch.load(opt.pretrain_model, map_location=str(setting.device)), strict=True)
#net.load_state_dict(torch.load(test_model_name, map_location=str(setting.device)), strict=True)
net = net.to(setting.device)

validate_MAE, validate_RMSE, validate_loss, time_cost, pred_mae, pred_mse  = estimator.evaluate(net, 1, eval_loader.__len__()*batch_eval_size)

print(validate_MAE, validate_RMSE)

