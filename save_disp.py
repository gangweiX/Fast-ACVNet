from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
import cv2
# cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Accurate and Efficient Stereo Matching via Attention Concatenation Volume (Fast-ACV)')
parser.add_argument('--model', default='Fast_ACVNet_plus', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--kitti15_datapath', default='/data/KITTI/KITTI_2015/', help='data path')
parser.add_argument('--kitti12_datapath', default='/data/KITTI/KITTI_2012/', help='data path')
parser.add_argument('--testlist', default='./filenames/kitti15_test.txt', help='testing list')
parser.add_argument('--loadckpt', default='',help='load the weights from a specific checkpoint')
parser.add_argument('--attention_weights_only', default=False, type=str,  help='only train attention weights')
# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.kitti15_datapath, args.kitti12_datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, args.attention_weights_only)
model = nn.DataParallel(model)
model.cuda()

#load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


save_dir = './output/kitti15/disp_0'


def test():
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            fn = os.path.join(save_dir, fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)
            # cv2.imwrite(fn, cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01),cv2.COLORMAP_JET))


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]


if __name__ == '__main__':
    test()
