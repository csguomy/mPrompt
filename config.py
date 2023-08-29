#import util.utils as util
import os
import torch

class config(object):
    def __init__(self, opt):
        self.unknown_folder = opt.unknown_folder
        self.min_mae = 10240000
        self.min_loss = 10240000
        self.dataset_name = opt.dataset_name
        self.lr = opt.lr
        self.batch_size = opt.batch_size
        self.eval_per_step = opt.eval_per_step
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.model_save_path = os.path.join(opt.checkpoints_dir, opt.name, opt.dataset_name) # path of saving model
        self.epoch = opt.max_epochs
        self.mode = opt.mode
        self.is_random_hsi = opt.is_random_hsi
        self.is_flip = opt.is_flip
        print(self.dataset_name)

        if self.dataset_name == 'SHA':
            self.eval_num = 182
            self.train_num = 300

            self.train_gt_map_path = '/userhome/Data/datasets/Processed_SHA_oriImg/den/train'
            self.eval_gt_map_path = '/userhome/Data/datasets/Processed_SHA_oriImg/den/test'
            self.train_img_path = '/userhome/Data/datasets/Processed_SHA_oriImg/ori/train_data/images'
            self.eval_img_path = '/userhome/Data/datasets/Processed_SHA_oriImg/ori/test_data/images'
            self.eval_gt_path = '/userhome/Data/datasets/Processed_SHA_oriImg/ori/test_data/ground_truth'
            self.box_path = ''

        elif self.dataset_name == 'SHB':
            self.eval_num = 316
            self.train_num = 400

            self.train_gt_map_path = '/userhome/Data/datasets/Processed_SHB_oriImg/den/train'
            self.eval_gt_map_path = '/userhome/Data/datasets/Processed_SHB_oriImg/den/test'
            self.train_img_path = '/userhome/Data/datasets/Processed_SHB_oriImg/ori/train_data/images'
            self.eval_img_path = '/userhome/Data/datasets/Processed_SHB_oriImg/ori/test_data/images'
            self.eval_gt_path = '/userhome/Data/datasets/Processed_SHB_oriImg/ori/test_data/ground_truth'
            self.box_path = ''
            
        elif self.dataset_name == 'QNRF':
            self.eval_num = 334
            self.train_num = 1201

            self.train_gt_map_path = '/userhome/Data/datasets/Processed_QNRF_large_oriImg/den/train'
            self.eval_gt_map_path = '/userhome/Data/datasets/Processed_QNRF_large_oriImg/den/test'
            self.train_img_path = '/userhome/Data/datasets/Processed_QNRF_large_oriImg/ori/train_data/images'
            self.eval_img_path = '/userhome/Data/datasets/Processed_QNRF_large_oriImg/ori/test_data/images'
            self.eval_gt_path = '/userhome/Data/datasets/Processed_QNRF_large_oriImg/ori/test_data/ground_truth'
            self.box_path = ''
            
        elif self.dataset_name == 'NWPU':
            self.eval_num = 500
            self.train_num = 3,109

            self.train_gt_map_path = '/userhome/Data/datasets/Processed_NWPU_large_oriImg/den/train'
            self.eval_gt_map_path = '/userhome/Data/datasets/Processed_NWPU_large_oriImg/den/test'
            self.train_img_path = '/userhome/Data/datasets/Processed_NWPU_large_oriImg/ori/train_data/images'
            self.eval_img_path = '/userhome/Data/datasets/Processed_NWPU_large_oriImg/ori/test_data/images'
            self.eval_gt_path = '/userhome/Data/datasets/Processed_NWPU_large_oriImg/ori/test_data/ground_truth'   
            self.box_path = '/userhome/Data/datasets/Ori_data/NWPU_data/jsons'

        # for extra images contain no gt counts
        elif self.dataset_name == 'Unknown':
            self.eval_num = 1500
            self.train_num = 1 # useless in fact

            self.train_gt_map_path = 'x'
            self.eval_gt_map_path = 'x'
            self.train_img_path = 'x'
            self.eval_img_path = self.unknown_folder
            self.eval_gt_path = 'x'
            self.box_path = ''
