from PIL import Image
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import random
import time
import scipy.io as scio
import h5py
import math
import glob
import os
from util import utils
import matplotlib.pyplot as plt
import cv2

class DatasetConstructor(data.Dataset):
    def __init__(self):
        self.datasets_com = ['SHA', 'SHB', 'QNRF', 'NWPU']
        return

    # return current dataset(SHA/SHB/QNRF) for current image
    def get_cur_dataset(self, img_name):
        check_list = [img_name.find(da) for da in self.datasets_com]
        check_list = np.array(check_list)
        cur_idx = np.where(check_list != -1)[0][0]
        return self.datasets_com[cur_idx]

    def resize(self, img, dataset_name, rand_scale_rate=0.0, perform_resize=True):
        height = img.size[1]
        width = img.size[0]
        resize_height = height
        resize_width = width

        if rand_scale_rate > 0.0:
            cur_rand = random.uniform(1-rand_scale_rate, 1+rand_scale_rate)
            resize_height = int(cur_rand * resize_height)
            resize_width = int(cur_rand * resize_width)

        if dataset_name == "SHA":
            sz = 416 # or 512
            if resize_height <= sz:
                tmp = resize_height
                resize_height = sz
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= sz:
                tmp = resize_width
                resize_width = sz
                resize_height = (resize_width / tmp) * resize_height
            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        elif dataset_name == "SHB":
            resize_height = height
            resize_width = width
        elif dataset_name.find("QNRF") != -1 or dataset_name.find("NWPU") != -1: # it is QNRF_large
            
            if resize_width >= 2048:
                tmp = resize_width
                resize_width = 2048
                resize_height = (resize_width / tmp) * resize_height

            if resize_height >= 2048:
                tmp = resize_height
                resize_height = 2048
                resize_width = (resize_height / tmp) * resize_width            
            
            if resize_height <= 512:
                tmp = resize_height
                resize_height = 512
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= 512:
                tmp = resize_width
                resize_width = 512
                resize_height = (resize_width / tmp) * resize_height

            # other constraints
            if resize_height < resize_width:
                if resize_width / resize_height > 2048/512: # the original is 512 instead of 416
                    resize_width = 2048
                    resize_height = 512
            else:
                if resize_height / resize_width > 2048/512:
                    resize_height = 2048
                    resize_width = 512

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF, NWPU")
        if perform_resize:
            img = transforms.Resize([resize_height, resize_width])(img)
            ratio_H = resize_height / height
            ratio_W = resize_width / width
            return img, ratio_H, ratio_W
        else:
            return resize_height, resize_width

class TrainDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 train_num,
                 data_dir_path,
                 gt_dir_path,
                 box_dir_path,
                 mode='crop',
                 dataset_name="JSTL",
                 device=None,
                 is_random_hsi=False,
                 is_flip=False,
                 fine_size = 400,
                 opt=None,
                 ):
        super(TrainDatasetConstructor, self).__init__()
        self.train_num = train_num
        self.opt = opt
        self.imgs = []
        self.fine_size = fine_size
        self.permulation = np.random.permutation(self.train_num)
        self.data_root, self.gt_root, self.box_root = data_dir_path, gt_dir_path, box_dir_path
        self.mode = mode
        self.device = device
        self.is_random_hsi = is_random_hsi
        self.is_flip = is_flip
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2))
        self.online_map = True if self.opt.rand_scale_rate > 0.0 else False
        # they are mapped as pairs
        imgs = sorted(glob.glob(self.data_root+'/*'))
        dens = sorted(glob.glob(self.gt_root+'/*'))
        self.train_num = len(imgs)
        print('Constructing training dataset...')
        for i in range(self.train_num):
            img_tmp = imgs[i]
            den = os.path.join(self.gt_root, os.path.basename(img_tmp)[:-4] + ".npy")
            if self.box_root != '':
                box = os.path.join(self.box_root, os.path.basename(img_tmp)[-8:-4] + ".json")
            else:
                print('The box file not exist :', self.box_root)
                box = ''
                
            assert den in dens, "Automatically generating density map paths corrputed!"
            self.imgs.append([imgs[i], den, box])
            
        
        ########## vscrowd dataset ##########
        # annotations-> videos -> frames
        frame_num = 0
        annotations_fold = '/userhome/Data/datasets/Ori_data/VSCrowd/annotations'
        annotations = sorted(glob.glob(annotations_fold+'/*'))
        videos_fold = '/userhome/Data/datasets/Ori_data/VSCrowd/videos/'
        assert len(annotations) == 634, print('annotations_fold: ', len(annotations)) 
        for video in annotations:
            frames=[]
            frame_samples=[]
            if video.find('train') != -1:
                video_name = os.path.basename(video)[:-4]
                
                with open(video, 'r') as f:
                    frames = f.readlines()
                    frame_samples = random.sample(frames, 2)
                f.close()
                    
            for frame in frame_samples:
                items = frame.split(' ')
                assert (len(items)-1)%7 == 0
                frameID = items[0]
                img_path = videos_fold + video_name + '/' + frameID + '.jpg'
                
                img = Image.open(img_path).convert("RGB")
                width, height = img.size
                if width < self.fine_size or height < self.fine_size:
                    continue
                    
                box = items[1:]
                self.imgs.append([img_path,'', box])
                frame_num += 1
        print('frame_num :', frame_num)
        ########## vscrowd dataset ##########        
        

        self.train_num = len(self.imgs)
        print(self.train_num)
        
        
    def __getitem__(self, index):
        if self.mode == 'crop':
      
            img_path, gt_map_path, box_path = self.imgs[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            cur_dataset='NWPU'
            ratio_h = 1.0
            ratio_w = 1.0
            
            if img_path.find('NWPU') != -1:
                cur_dataset = super(TrainDatasetConstructor, self).get_cur_dataset(img_path)
                img, ratio_h, ratio_w = super(TrainDatasetConstructor, self).resize(img, cur_dataset, self.opt.rand_scale_rate)
            
            width, height = img.size
            gt_map = np.zeros([height, width], dtype=np.float32)
            
            box_one = np.ones([height, width], dtype=np.float32)
            box_mask = np.zeros([height, width], dtype=np.float32)
            box_num = np.zeros([height, width], dtype=np.float32)
            seg_map = np.zeros([2, height, width], dtype=np.float32)
            
            #print(img_path)
            if img_path.find('NWPU') != -1:
                
                # The ground-truth of density map
                gt_map = np.squeeze(np.load(gt_map_path).astype(np.float32))
          
                # generate box mask
                if box_path != '':
                    #box_mask = np.zeros([height, width], dtype=np.float32)
                    #seg_map = np.zeros([2, height, width], dtype=np.float32)
                    
                    with open(box_path, 'r') as f:
                        result = json.load(f)
                        points_box = np.array(result['boxes'])
                    f.close()

                    for b in points_box:
                        x_min = math.floor(b[0]*ratio_w)
                        y_min = math.floor(b[1]*ratio_h)
                        x_max = math.ceil(b[2]*ratio_w)
                        y_max = math.ceil(b[3]*ratio_h)
                        #print(x_min, y_min, x_max, y_max)
                        box_mask[y_min:y_max, x_min:x_max] = 1.0
                    
                        box_num[y_min:y_max, x_min:x_max] += 1.0
                        seg_map[0][y_min:y_max, x_min:x_max] += (x_max-x_min)
                        seg_map[1][y_min:y_max, x_min:x_max] += (y_max-y_min)
            
            box_num = box_one - box_mask + box_num
            seg_map = seg_map / box_num
            
            
            
            # DataAugmentation
            gt_map = Image.fromarray(gt_map)
            box_mask = Image.fromarray(np.squeeze(box_mask).astype(np.float32)) 
            seg_map = np.squeeze(seg_map).astype(np.float32) 
            if self.is_random_hsi:
                img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            if self.is_flip:
                flip_random = random.random()
                if flip_random > 0.5:
                    img = F.hflip(img)
                    gt_map = F.hflip(gt_map)
                    box_mask = F.hflip(box_mask)
                    #fbs = F.hflip(fbs)
                    seg_map = np.flip(seg_map, axis=2)

            img, gt_map, seg_map = transforms.ToTensor()(np.array(img)), torch.from_numpy(np.array(gt_map)).view(1, height, width), torch.from_numpy(seg_map.copy()).view(2, height, width)

            img_shape = img.shape  # C, H, W
            gt_map_shape = gt_map.shape
            #gt_seg_shape = fbs.shape
            gt_seg_shape = seg_map.shape
            
          
            # also scale gt_map
            if img_shape[1] != gt_map_shape[1] or img_shape[2] != gt_map_shape[2] or img_shape[1] != gt_seg_shape[1] or img_shape[2] != gt_seg_shape[2]:
                print(img_shape, gt_map_shape, gt_seg_shape)
                assert 1==2
                #gt_map = functional.interpolate(gt_map.unsqueeze(0), (img_shape[1], img_shape[2]), mode='bilinear').squeeze(0)

            rh, rw = random.randint(0, img_shape[1] - self.fine_size), random.randint(0, img_shape[2] - self.fine_size)
            p_h, p_w = self.fine_size, self.fine_size
            img = img[:, rh:rh + p_h, rw:rw + p_w]
            gt_map = gt_map[:, rh:rh + p_h, rw:rw + p_w]
            seg_map = seg_map[:, rh:rh + p_h, rw:rw + p_w]
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
          
            return img.view(3, self.fine_size, self.fine_size), gt_map.view(1, self.fine_size, self.fine_size), seg_map.view(2, self.fine_size, self.fine_size), os.path.basename(img_path)
           

    def __len__(self):
        return self.train_num

# For evalation
class EvalDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 validate_num,
                 data_dir_path,
                 gt_dir_path,
                 box_dir_path,
                 mode="crop",
                 dataset_name="JSTL",
                 device=None,
                 ):
        super(EvalDatasetConstructor, self).__init__()
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.mode = mode
        self.device = device
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2))
        # they are mapped as pairs
        imgs = sorted(glob.glob(self.data_root+'/*'))
        dens = sorted(glob.glob(self.gt_root+'/*'))
        self.validate_num = len(imgs)
        print('Constructing testing dataset...')
        print(self.validate_num)

        self.extra_dataset = False
        if self.dataset_name == 'Unknown':
            self.extra_dataset = True

        for i in range(self.validate_num):
            img_tmp = imgs[i]

            if self.extra_dataset == False:
                den = os.path.join(self.gt_root, os.path.basename(img_tmp)[:-4] + ".npy")
                assert den in dens, "Automatically generating density map paths corrputed!"
                self.imgs.append([imgs[i], den])
            else:
                self.imgs.append(imgs[i])

    def __getitem__(self, index):
        if self.mode == 'crop':
            if self.extra_dataset:
                img_path = self.imgs[index]
            else:
                img_path, gt_map_path = self.imgs[index]

            img = Image.open(img_path).convert("RGB")
            cur_dataset = super(EvalDatasetConstructor, self).get_cur_dataset(img_path)
            img, ratio_h, ratio_w = super(EvalDatasetConstructor, self).resize(img, cur_dataset)
            width, height = img.size
            
            img = transforms.ToTensor()(img)
            img_resized = img
            img_shape = img.shape
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            
        
            # crops for testing
            patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
            imgs = []
            for i in range(3):
                for j in range(3):
                    start_h, start_w = (patch_height // 2) * i, (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])

            imgs = torch.stack(imgs)

            # ------for density maps------#
            patch_h, patch_w = imgs.size(2), imgs.size(3)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path).astype(np.float32)))
            gt_map = transforms.ToTensor()(np.array(gt_map))
            gt_shape = gt_map.shape

            gt_H, gt_W = gt_shape[1], gt_shape[2]

            return img_path, img, gt_map.view(1, gt_H, gt_W), torch.tensor(gt_H), torch.tensor(gt_W), torch.tensor(patch_h), torch.tensor(patch_w)

    def __len__(self):
        return self.validate_num
