import timm
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from torch.optim import lr_scheduler
import functools

from net.HRNet.reg_seg import HighResolutionNet as reg_seg

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, opt):
     """Return a learning rate scheduleR
     Parameters:
         optimizer          -- the optimizer of the network
         opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
                               opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
     For 'linear', we keep the same learning rate for the first <opt.niter> epochs
     and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
     For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
     See https://pytorch.org/docs/stable/optim.html for more details.
     """
     if opt.lr_policy == 'linear':
         def lambda_rule(epoch):
             lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
             return lr_l
         scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
     elif opt.lr_policy == 'step':
         scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.4) # 5e-5 -> 2e-5
     elif opt.lr_policy == 'plateau':
         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
     elif opt.lr_policy == 'cosine':
         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=opt.min_lr)
     else:
         return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
     return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def select_optim(net, opt):
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
    elif opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(net.parameters(), opt.lr, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError('This optimizer has not implemented yet')
    return optimizer


def init_net(net, init_type='normal', init_gain=0.01, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # later using custom init, for now, just init inside.
    # init_weights(net, init_type, init_gain=init_gain)
    return net

def define_net(opt):
    net_name = opt.net_name
    print('net_name: ', net_name)

    if net_name == 'reg_seg':
        net = reg_seg()
        #net.init_weights(pretrained = opt.pretrain_model)
    else:
        raise NotImplementedError('Unrecognized model: '+ net_name)
    return net


