from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import random
import math
import glob
import os
from util import utils

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
            return img, resize_height, resize_width, ratio_H, ratio_W
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
        self.data_root, self.gt_root = data_dir_path, gt_dir_path
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
            assert den in dens, "Automatically generating density map paths corrputed!"
            self.imgs.append([imgs[i], den])

        self.train_num = len(self.imgs)
        print(self.train_num)
        

    def __getitem__(self, index):
        if self.mode == 'crop':
      
            img_path, gt_map_path = self.imgs[index]
            
            #print(img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            cur_dataset = super(TrainDatasetConstructor, self).get_cur_dataset(img_path)

            img, resize_height, resize_width, ratio_h, ratio_w = super(TrainDatasetConstructor, self).resize(img, cur_dataset, self.opt.rand_scale_rate)
            width, height = img.size
            
            # The ground-truth of density map
            gt_map = np.squeeze(np.load(gt_map_path).astype(np.float32))
            gt_map_sum = math.ceil(np.sum(gt_map))
            mask_den = (gt_map > 0)
            
            ## get mask from pretrained segmentation task
            mask_seg_path = gt_map_path.replace('den', 'mask')
            mask = np.squeeze(np.load(mask_seg_path).astype(np.float32))
            mask = (mask > 0.055)
            mask_seg = mask

            mat_name = img_path.replace('images', 'ground_truth')[:-4] + ".mat"
            points = scio.loadmat(mat_name)['annPoints']
            points_sum = len(points)
            
            # generate bigger kernal gaussian mask than density map as segmentation mask
            if points_sum > 0:
                h = resize_height
                w = resize_width
                for idx, p in enumerate(points):
                    p = np.round(p).astype(int)
                    p[1], p[0] = min(h-1, math.floor(p[1] * ratio_h)), min(w-1, math.floor(p[0] * ratio_w))
                    points[idx] = (p[0], p[1])
                
                mask_gaussian = generate_density_map(shape=(height, width),points=points,boxes=[],f_sz=35,sigma=4)
                mask_gaussian = np.reshape(mask_gaussian, [height, width])  # transpose into w, h
                mask_gaussian = np.int32(mask_gaussian > 0)  
                
                # Add mask_gaussian blob to pesudo seg mask
                #mask = np.bitwise_or(mask_gaussian, mask)
                
                # baseline of dot-segmentation 
                #mask = mask_gaussian
                
            else:
                print('points_sum==0 ', img_path)
                          
            # Resize: annotation label vision
            mask_annotation = np.zeros((img_vision.size[1], img_vision.size[0]), np.uint8)
            for i in range(0, points_sum):
                mask_annotation = cv2.circle(mask_annotation, (math.ceil(points[i][0]), math.ceil(points[i][1])), 1, 1, -1) 
            

            # KNN mask
            k_n = 3
            points_sum = len(points)
            mask_knn = np.zeros((img.size[1], img.size[0]), np.uint8)
            if points_sum >= k_n+1:
                points_knn = KNN(points, k_n)            
                points_knn_sum = len(points_knn)
        
                for i in range(0, points_knn_sum):
                    k = k_n
                    r = points_knn[i][k-1]
                    r_c = np.mean(points_knn[i][0:k-1])
                    while r > r_c * 2:
                        k = k-1
                        r = points_knn[i][k-1]
                        if points_knn[i][0:k-1]:
                            r_c = np.mean(points_knn[i][0:k-1])
                        else:
                            r_c = r
                    mask_knn = cv2.circle(mask_knn, (math.ceil(points[i][0]), math.ceil(points[i][1])), math.floor(r), 1, -1) 
                 
                # operation: and 
                mask = np.bitwise_and(mask_knn, mask) 
            
            else:
                mask_knn = np.ones((img.size[1], img.size[0]), np.uint8)
                print('points_sum < k+1 ', points_sum, img_path)            
                      
         
            # or
            mask = np.bitwise_or(mask_den, mask)

            # DataAugmentation
            # form numpy to PIL
            mask = Image.fromarray(mask)
            gt_map = Image.fromarray(gt_map)
            mask_knn = Image.fromarray(mask_knn)
            if self.is_random_hsi:
                #img = img
                img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            if self.is_flip:
                flip_random = random.random()
                if flip_random > 0.5:
                    img = F.hflip(img)
                    gt_map = F.hflip(gt_map)
                    mask = F.hflip(mask)
                    mask_knn = F.hflip(mask_knn)
                                    
            # PIL crop
            # from PIL to tensor
            img, gt_map, mask, mask_knn = transforms.ToTensor()(np.array(img)), torch.from_numpy(np.array(gt_map)).view(1, height, width), torch.from_numpy(np.array(mask)).view(1, height, width), torch.from_numpy(np.array(mask_knn)).view(1, height, width)
            
            img_shape = img.shape  # C, H, W
            gt_map_shape = gt_map.shape
            gt_seg_shape = mask.shape
       
            # also scale gt_map
            if img_shape[1] != gt_map_shape[1] or img_shape[2] != gt_map_shape[2] or img_shape[1] != gt_seg_shape[1] or img_shape[2] != gt_seg_shape[2]:
                print(img.shape, gt_map.shape, mask.shape)
                assert 1==2
                gt_map = functional.interpolate(gt_map.unsqueeze(0), (img_shape[1], img_shape[2]), mode='bilinear').squeeze(0)
            
            # crop
            rh, rw = random.randint(0, img_shape[1] - self.fine_size), random.randint(0, img_shape[2] - self.fine_size)
            p_h, p_w = self.fine_size, self.fine_size

            img = img[:, rh:rh + p_h, rw:rw + p_w]
            gt_map = gt_map[:, rh:rh + p_h, rw:rw + p_w]
            mask = mask[:, rh:rh + p_h, rw:rw + p_w]
            mask_knn = mask_knn[:, rh:rh + p_h, rw:rw + p_w]

            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            mask = (mask > 0).float()
            return img.view(3, self.fine_size, self.fine_size), gt_map.view(1, self.fine_size, self.fine_size), mask.view(1, self.fine_size, self.fine_size), mask_knn.view(1, self.fine_size, self.fine_size), os.path.basename(img_path)

    def __len__(self):
        return self.train_num

# For evalation, we also return img_path.
# This help get the paths of '.mat' recording the real num(not from density map).
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


        # rank
        if self.extra_dataset == False:
            self.imgs_new = []
            self.cal_load_list = torch.zeros(self.validate_num)
            print('Pre-reading the resized image size info, and sort... please wait for round 1 min')

            for i in range(self.validate_num):
                img_path, _ = self.imgs[i]
                img = Image.open(img_path).convert("RGB")
                cur_dataset = super(EvalDatasetConstructor, self).get_cur_dataset(img_path)
                # do not resize, just get the resized size to acceralate
                H, W = super(EvalDatasetConstructor, self).resize(img, cur_dataset, perform_resize=False)
                cal_load =  H*W
                self.cal_load_list[i] = cal_load

            # sort the img_path in a descending order acoorindg the cal_load
            new_load_list, indices = torch.sort(self.cal_load_list, descending=True)
            for i in range(self.validate_num):
                cur_index = indices[i]
                # select img_path-den_path pair from self.imgs to form a new imgs_new list sorted by cal_load
                self.imgs_new.append(self.imgs[cur_index])

            # finally, rename self.imgs_new to self.imgs
            self.imgs = self.imgs_new        
        


    def __getitem__(self, index):
        if self.mode == 'crop':
            if self.extra_dataset:
                img_path = self.imgs[index]
            else:
                img_path, gt_map_path = self.imgs[index]

            # ---------for image---------#
            img = Image.open(img_path).convert("RGB")
            if self.extra_dataset:
                cur_dataset = 'NWPU' # using QNRF is fine for unknown
            else:
                cur_dataset = super(EvalDatasetConstructor, self).get_cur_dataset(img_path)
                
            # resize    
            img, resize_height, resize_width, ratio_h, ratio_w = super(EvalDatasetConstructor, self).resize(img, cur_dataset)
            img = transforms.ToTensor()(img)
            img_resized = img
            
            # shape
            img_shape = img.shape
            
            if self.extra_dataset:
                mask = torch.from_numpy(np.zeros((1, img_shape[1], img_shape[2]), dtype=float))
                gt_map = torch.from_numpy(np.zeros((1, img_shape[1], img_shape[2]), dtype=float))
                
            else:
                #---from segmentation mask---#
                mask_map_path = gt_map_path.replace('den', 'mask')
                mask = Image.fromarray(np.squeeze(np.load(mask_map_path).astype(np.float32)))
                mask = torch.from_numpy(np.array(mask)).view(1, img_shape[1], img_shape[2])
                # ------for density maps-----#
                gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path).astype(np.float32)))
                gt_map = torch.from_numpy(np.array(gt_map)).view(1, img_shape[1], img_shape[2])
                
            # shape    
            mask_shape = mask.shape
            gt_shape = gt_map.shape     
            # validate shape
            if img_shape[1] != mask_shape[1] or img_shape[2] != mask_shape[2] or img_shape[1] != gt_shape[1] or img_shape[2] != gt_shape[2]:
                print(img_shape, mask_shape, gt_shape)
                assert 1==2
                
            
            # -----crop image & mask-----#
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
            imgs = []
            for i in range(3):
                for j in range(3):
                    start_h, start_w = (patch_height // 2) * i, (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])

            imgs = torch.stack(imgs)
            patch_h, patch_w = imgs.size(2), imgs.size(3)
            # -----crop image & mask-----#                
            
            gt_map = torch.ones(1, img_shape[1], img_shape[2])
            #print(gt_map.shape, img_shape)
            
            #---from segmentation mask---#
            #mask = functional.conv2d(mask.view(1, *(mask_shape)), self.kernel, bias=None, stride=2, padding=0) # [c, h, w] -> [1, c, h/2, w/2]
            mask = (mask > 0).float()
            mask_shape = mask.shape
            mask = mask.view(1, mask_shape[1], mask_shape[2])
            
            
            # -----crop image & mask-----#
            #mask_height, mask_width = (mask_shape[2]) // 2, (mask_shape[3]) // 2
            mask_height, mask_width = (mask_shape[1]) // 2, (mask_shape[2]) // 2
            masks = []
            #print('mask: ', mask_shape)
            for i in range(3):
                for j in range(3):
                    start_h, start_w = (mask_height // 2) * i, (mask_width // 2) * j
                    masks.append(mask[:, start_h:start_h + mask_height, start_w:start_w + mask_width])

            masks = torch.stack(masks) 
            mask_H, mask_W = masks.size(2), masks.size(3)             
            
            # ------for density maps------#
            #gt_map = functional.conv2d(gt_map.view(1, *(gt_shape)), self.kernel, bias=None, stride=2, padding=0)
            #gt_H, gt_W = gt_shape[1]//2, gt_shape[2]//2
            gt_H, gt_W = gt_shape[1], gt_shape[2]

            return img_shape, img_path, imgs, masks, gt_map.view(1, gt_H, gt_W), torch.tensor(gt_H), torch.tensor(gt_W), torch.tensor(patch_h), torch.tensor(patch_w), torch.tensor(mask_H), torch.tensor(mask_W)
            
            
    def __len__(self):
        return self.validate_num




