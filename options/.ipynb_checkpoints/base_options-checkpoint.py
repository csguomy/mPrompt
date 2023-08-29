import argparse
import os
import torch
#import util.utils as util
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

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # unknown_video/frames1_full_img, unknown_video/frames2_full_img; we only know gt counts for each frame
        parser.add_argument('--model_ema', default=0, type=int, help='1 or 0')
        parser.add_argument('--model_ema_force_cpu', default=0, help='1 or 0')
        parser.add_argument('--model_ema_decay', default=0.996, type=float, help='1 or 0')
        parser.add_argument('--use-background', type=bool, default=True, help='whether to use background modelling')
        parser.add_argument('--unknown_folder', default='unknown_video/frames2_full_img', help='test_unknown, unknown_video/frames1_img， unknown_video/frames2_img')
        parser.add_argument('--dataset_name', default='JSTL_large', help='SHA|SHB|QNRF|NWPU')
        parser.add_argument('--rand_scale_rate', type=float, default=0.0, help='if 0.2, means 0.8-1.2')
        parser.add_argument('--pretrain', type=int, default=0, help='1|0')
        parser.add_argument('--eval_size_per_GPU', type=int, default=1, help='...')
        parser.add_argument('--pretrain_model', type=str, default='', help='path of pretrained model')
        parser.add_argument('--base_mae', type=str, default='60,10,100', help='something like, 60, 10, 100')
        parser.add_argument('--multi_head', type=int, default=2, help='1|2|...|')
        parser.add_argument('--cls_num', type=int, default=1, help='4|5|6..., number of domain')
        parser.add_argument('--drop_rate', type=float, default=0.3, help='0.3')
        parser.add_argument('--batch_size', type=int, default=5, help='input batch size')
        parser.add_argument('--cls_w', type=float, default=1., help='weight of cls')
        parser.add_argument('--net_name', type=str, default='hrnet_aspp_relu_s6', help='res_unet|res_unet_leaky')
        parser.add_argument('--fine_size', type=int, default=400, help='cropped size')
        parser.add_argument('--name', type=str, default='hrnet_aspp_relu', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2')
        parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./output', help='models are saved here')
        parser.add_argument('--mode', type=str, default='crop', help='crop')
        parser.add_argument('--is_flip', type=int, default=1, help='whether perform flipping data augmentation')
        parser.add_argument('--is_random_hsi', type=int, default=1, help='whether perform random hsi data augmentation')
        parser.add_argument('--is_color_aug', type=int, default=1, help='whether perform color augmentation')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer [sgd|adam|adamW]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.01, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay of adam')
        parser.add_argument('--amsgrad', type=int, default=0, help='weight using amsgrad of adam')
        parser.add_argument('--eval_per_step', type=int, default=float("inf"), help='When detailed change super-parameter, may need it, step of evaluation')
        parser.add_argument('--eval_per_epoch', type=int, default=1, help='epoch step of evaluation')
        parser.add_argument('--start_eval_epoch', type=int, default=100, help='beginning epoch of evaluation')
        parser.add_argument('--print_step', type=int, default=10, help='print step of loss')
        parser.add_argument('--max_epochs', type=int, default=500, help='Epochs of training')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau|cosine')
        
        # lr_policy for cosine
        parser.add_argument('--niter', type=int, default=1, help='niter_per_ep for cosine_scheduler')
        parser.add_argument('--min_lr', type=float, default=1e-7, help='minimal learning rate for cosine_scheduler')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs of training ')
        parser.add_argument('--warmup_steps', type=int, default=0, help='Warmup step of training')
        
        parser.add_argument('--lr_decay_iters', type=int, default=300, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--test_eval', type=int, default=0, help='whether test only')

        # distributed args
        parser.add_argument('--dist_eval', default=0, type=int, help='useless in fact')
        parser.add_argument('--pin_mem', default=1, type=int, help='whether using pin_memory for dataloader')
        parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

        parser.add_argument('--device', default='cuda',  help='device to use for training / testing')
        parser.add_argument('--seed', type=int, default=0, help='seed')
        parser.add_argument('--downsample_ratio', type=int, default=8, help='downsample ratio')
        parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')
        parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')
        parser.add_argument('--reg', type=float, default=10.0, help='entropy regularization in sinkhorn')
        parser.add_argument('--norm_cood', type=int, default=0, help='whether to norm cood when computing distance')
        parser.add_argument('--num_of_iter_in_ot', type=int, default=100, help='sinkhorn iterations')

        self.initialized = True
        return parser

    def gather_options(self, options=None):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)


        self.parser = parser
        if options == None:
            return parser.parse_args()
        else:
            return parser.parse_args(options)

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.dataset_name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, options=None):

        opt = self.gather_options(options=options)
        opt.isTrain = self.isTrain   # train or test


        self.print_options(opt)

        # set gpu ids
        os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # re-order gpu ids
        opt.gpu_ids = [i.item() for i in torch.arange(len(opt.gpu_ids))]
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
