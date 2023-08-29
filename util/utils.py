from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import numpy as np
import scipy
import math
from PIL import Image


def show(origin_map, gt_map, predict, index):
    figure, (origin, gt, pred) = plt.subplots(1, 3, figsize=(20, 4))
    origin.imshow(origin_map)
    origin.set_title("origin picture")
    gt.imshow(gt_map, cmap=plt.cm.jet)
    gt.set_title("gt map")
    pred.imshow(predict, cmap=plt.cm.jet)
    pred.set_title("prediction")
    plt.suptitle(str(index) + "th sample")
    plt.show()
    plt.close()

class ColorAugmentation(object):
    def __init__(self):
        self.eig_vec = torch.Tensor([
             [0.4009, 0.7192, -0.5675],
             [-0.8140, -0.0045, -0.5808],
             [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


class HSI_Calculator(nn.Module):
    def __init__(self):
        super(HSI_Calculator, self).__init__()

    def forward(self, image):
        image = transforms.ToTensor()(image)
        I = torch.mean(image)
        Sum = image.sum(0)
        Min = 3 * image.min(0)[0]
        S = (1 - Min.div(Sum.clamp(1e-6))).mean()
        numerator = (2 * image[0] - image[1] - image[2]) / 2
        denominator = ((image[0] - image[1]) ** 2 + (image[0] - image[2]) * (image[1] - image[2])).sqrt()
        theta = (numerator.div(denominator.clamp(1e-6))).clamp(-1 + 1e-6, 1 - 1e-6).acos()
        logistic_matrix = (image[1] - image[2]).ceil()
        H = (theta * logistic_matrix + (1 - logistic_matrix) * (360 - theta)).mean() / 360
        return H, S, I


def eval_steps_adaptive(var):
    return {
            400 * 100: 5000,
            400 * 500: 2000,
            400 * 1000: 1000,
    }.get(var, 1600)


def get_density_map_gaussian(H, W, ratio_h, ratio_w, points, gaussian_radius=7, sigma=4):
    h = H
    w = W
    density_map = np.zeros([h, w], dtype=np.float32)
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, math.floor(p[1] * ratio_h)), min(w-1, math.floor(p[0] * ratio_w))
      
        sigma = max(1, sigma)

        #gaussian_radius = 7
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < 0 or p[0] < 0:
            continue
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(h, p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(w, p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    return density_map

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image -= np.array([123.675, 116.28, 103.53], np.float32)
    image /= np.array([58.395, 57.12, 57.375], np.float32)
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(phi, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'b0' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b0_backbone_weights.pth",
        'b1' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b1_backbone_weights.pth",
        'b2' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b2_backbone_weights.pth",
        'b3' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b3_backbone_weights.pth",
        'b4' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b4_backbone_weights.pth",
        'b5' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b5_backbone_weights.pth",
    }
    url = download_urls[phi]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
