from PIL import Image
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

    def resize(self, img, dataset_name):
        height = img.size[1]
        width = img.size[0]
        resize_height = height
        resize_width = width
        if dataset_name == "SHA":
            if resize_height <= 416:
                tmp = resize_height
                resize_height = 416
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= 416:
                tmp = resize_width
                resize_width = 416
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
                if resize_width / resize_height > 4: # 1024/416=2.46
                    resize_width = 2048
                    resize_height = 512
            else:
                if resize_height / resize_width > 4:
                    resize_height = 2048
                    resize_width = 512

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        img = transforms.Resize([resize_height, resize_width])(img)
        return img


class TrainDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 train_num,
                 data_dir_path,
                 gt_dir_path,
                 mode='crop',
                 dataset_name="JSTL",
                 device=None,
                 is_random_hsi=False,
                 is_flip=False,
                 fine_size = 400
                 ):
        super(TrainDatasetConstructor, self).__init__()
        self.train_num = train_num
        self.imgs = []
        self.fine_size = fine_size
        self.permulation = np.random.permutation(self.train_num)
        self.data_root, self.gt_root = data_dir_path, gt_dir_path
        self.mode = mode
        self.device = device
        self.is_random_hsi = is_random_hsi
        self.is_flip = is_flip
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2))
        # they are mapped as pairs
        imgs = sorted(glob.glob(self.data_root+'/*'))
        dens = sorted(glob.glob(self.gt_root+'/*'))
        self.train_num = len(imgs)
        print('Constructing training dataset...')
        for i in range(self.train_num):
            img_tmp = imgs[i]
            den = os.path.join(self.gt_root, os.path.basename(img_tmp)[:-4] + ".npy")
            assert den in dens, "Automatically generating density map paths corrputed!"
            self.imgs.append([imgs[i], den])

    def __getitem__(self, index):
        if self.mode == 'crop':
            img_path, gt_map_path = self.imgs[index]
            # get domain
            class_id = 0
            if os.path.basename(img_path).find('SHA') != -1:
                class_id = 0
            elif os.path.basename(img_path).find('SHB') != -1:
                class_id = 1
            elif os.path.basename(img_path).find('QNRF') != -1:
                class_id = 2
            elif os.path.basename(img_path).find('NWPU') != -1:
                class_id = 3
            else:
                assert 1==2

            class_id = torch.tensor(class_id).long()
            img = Image.open(img_path).convert("RGB")
            cur_dataset = super(TrainDatasetConstructor, self).get_cur_dataset(img_path)
            img = super(TrainDatasetConstructor, self).resize(img, cur_dataset)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path).astype(np.float32)))

            if self.is_random_hsi:
                img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            if self.is_flip:
                flip_random = random.random()
                if flip_random > 0.5:
                    img = F.hflip(img)
                    gt_map = F.hflip(gt_map)

            img, gt_map = transforms.ToTensor()(np.array(img)), transforms.ToTensor()(np.array(gt_map))
            img_shape = img.shape  # C, H, W
            rh, rw = random.randint(0, img_shape[1] - self.fine_size), random.randint(0, img_shape[2] - self.fine_size)
            p_h, p_w = self.fine_size, self.fine_size
            img = img[:, rh:rh + p_h, rw:rw + p_w]
            gt_map = gt_map[:, rh:rh + p_h, rw:rw + p_w]
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            gt_map = functional.conv2d(gt_map.view(1, 1, self.fine_size, self.fine_size), self.kernel, bias=None, stride=2, padding=0)
            return img.view(3, self.fine_size, self.fine_size), gt_map.view(1, 200, 200), class_id, os.path.basename(img_path)

    def __len__(self):
        return self.train_num


#
# For evalation, we also return img_path.
# This help get the paths of '.mat' recording the real num(not from density map).
class EvalDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 validate_num,
                 data_dir_path,
                 gt_dir_path,
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
        imgs = glob.glob(self.data_root+'/*')
        img_last_index = imgs[0].rfind('_')
        imgs = sorted(imgs, key=lambda name: int(name[img_last_index+1:-4]))
        self.validate_num = len(imgs)
        print(self.validate_num)
        print('Constructing testing dataset...')

        self.extra_dataset = False
        if self.dataset_name.find('unknown') != -1:
            self.extra_dataset = True

        for i in range(self.validate_num):
            img_tmp = imgs[i]
            if self.extra_dataset == False:
                dens = glob.glob(self.gt_root+'/*')
                den_last_index = dens[0].rfind('_')
                dens = sorted(dens, key=lambda name: int(name[den_last_index+1:-4]))
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

            # get domain
            class_id = 0
            if os.path.basename(img_path).find('SHA') != -1:
                class_id = 0
            elif os.path.basename(img_path).find('SHB') != -1:
                class_id = 1
            elif os.path.basename(img_path).find('QNRF') != -1:
                class_id = 2
            elif os.path.basename(img_path).find('NWPU') != -1:
                class_id = 3
            else:
                class_id = -1

            class_id = torch.tensor(class_id).long()

            img = Image.open(img_path).convert("RGB")

            if self.extra_dataset:
                cur_dataset = 'QNRF' # using QNRF is fine for unknown
            else:
                cur_dataset = super(EvalDatasetConstructor, self).get_cur_dataset(img_path)
            img = super(EvalDatasetConstructor, self).resize(img, cur_dataset)
            img = transforms.ToTensor()(img)
            img_resized = img
            img_shape = img.shape
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
            imgs = []
            for i in range(3):
                for j in range(3):
                    start_h, start_w = (patch_height // 2) * i, (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])

            imgs = torch.stack(imgs)

            if self.extra_dataset: # unknown dataset, return these things are enough
                return img_resized, img_path, imgs, np.array(img_shape) # return 'img_shape' for obtain 'fake' density map size when evaluating

            # ------       for density maps
            patch_h, patch_w = imgs.size(2), imgs.size(3)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path).astype(np.float32)))
            gt_map = transforms.ToTensor()(np.array(gt_map))
            gt_shape = gt_map.shape
            gt_map = functional.conv2d(gt_map.view(1, *(gt_shape)), self.kernel, bias=None, stride=2, padding=0)
            gt_H, gt_W = gt_shape[1]//2, gt_shape[2]//2
            return img_path, imgs, gt_map.view(1, gt_H, gt_W), class_id, torch.tensor(gt_H), torch.tensor(gt_W), torch.tensor(patch_h), torch.tensor(patch_w)

    def __len__(self):
        return self.validate_num



'''
It is totally deprecated. You can delete it anytime.
'''
class EvalDatasetConstructor_old(DatasetConstructor):
    def __init__(self,
                 validate_num,
                 data_dir_path,
                 gt_dir_path,
                 mode="crop",
                 dataset_name="JSTL",
                 device=None,
                 ):
        super(EvalDatasetConstructor_old, self).__init__()
        self.validate_num = validate_num
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
        # SHB_IMG_73.jpg --> SHB_GT_IMG_73.npy
        # QNRF_img_0002.jpg --> QNRF_GT_IMG_2.npy (only QNRF has 'img'(not IMG) and have prefix 0s)
        print('Constructing testing dataset...')
        for i in range(self.validate_num):
            img_tmp = imgs[i]
            # construct den path
            img_str = 'img' if 'img' in img_tmp else 'IMG'
            den_tmp = img_tmp.replace(img_str, 'GT_IMG').replace('.jpg', '.npy')
            # remove prefix 0s
            den = '_'.join(den_tmp.split('_')[:-1] + [str(int(den_tmp.split('_')[-1][:-4])) + '.npy'])
            den = os.path.join(self.gt_root, den.split('/')[-1])
            assert den in dens, "Automatically generating density map paths corrputed!"
            self.imgs.append([imgs[i], den, i+1])

    def __getitem__(self, index):
        if self.mode == 'crop':
            img_path, gt_map_path, img_index = self.imgs[index]

            class_id = 0
            if os.path.basename(img_path).find('SHA') != -1:
                class_id = 0
            elif os.path.basename(img_path).find('SHB') != -1:
                class_id = 1
            elif os.path.basename(img_path).find('QNRF') != -1:
                class_id = 2
            elif os.path.basename(img_path).find('NWPU') != -1:
                class_id = 3
            else:
                assert 1==2
            class_id = torch.tensor(class_id).long()

            img = Image.open(img_path).convert("RGB")
            cur_dataset = super(EvalDatasetConstructor_old, self).get_cur_dataset(img_path)
            img = super(EvalDatasetConstructor_old, self).resize(img, cur_dataset)
            img = transforms.ToTensor()(img)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path).astype(np.float32)))
            gt_map = transforms.ToTensor()(np.array(gt_map))

            img_shape, gt_shape = img.shape, gt_map.shape  # C, H, W
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
            imgs = []
            for i in range(3):
                for j in range(3):
                    start_h, start_w = (patch_height // 2) * i, (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])

            imgs = torch.stack(imgs)
            gt_map = functional.conv2d(gt_map.view(1, *(gt_shape)), self.kernel, bias=None, stride=2, padding=0)
            return img_path, img_index, imgs, gt_map.view(1, gt_shape[1] // 2, gt_shape[2] // 2), class_id

    def __len__(self):
        return self.validate_num
