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
from tqdm import tqdm, trange
from datasets import middlebury_loader as mb
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
import cv2
from PIL import Image
from torchvision import transforms
from datasets import readpfm as rp
# cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Accurate and Efficient Stereo Matching via Attention Concatenation Volume (Fast-ACV)')
parser.add_argument('--model', default='Fast_ACVNet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default="/data/Middlebury/", help='data path')
parser.add_argument('--resolution', type=str, default='H')
parser.add_argument('--loadckpt', default='./pretrained_models/Fast_ACV/generalization.ckpt',help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()
train_limg, train_rimg, train_gt, test_limg, test_rimg = mb.mb_loader(args.datapath, res=args.resolution)
model = __models__[args.model](args.maxdisp, False)
model = nn.DataParallel(model)
model.cuda()
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])
model.eval()

os.makedirs('./demo-imgs/', exist_ok=True)

def test_trainset():
    op = 0
    mae = 0

    for i in trange(len(train_limg)):

        limg_path = train_limg[i]
        rimg_path = train_rimg[i]

        limg = Image.open(limg_path).convert('RGB')
        rimg = Image.open(rimg_path).convert('RGB')

        w, h = limg.size
        wi, hi = (w // 32 + 1) * 32, (h // 32 + 1) * 32

        limg = limg.crop((w - wi, h - hi, w, h))
        rimg = rimg.crop((w - wi, h - hi, w, h))

        limg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
        rimg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
        limg_tensor = limg_tensor.unsqueeze(0).cuda()
        rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

        with torch.no_grad():
            pred_disp  = model(limg_tensor, rimg_tensor)[-1]

            pred_disp = pred_disp[:, hi - h:, wi - w:]

        pred_np = pred_disp.squeeze().cpu().numpy()

        torch.cuda.empty_cache()

        disp_gt, _ = rp.readPFM(train_gt[i])
        disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
        disp_gt[disp_gt == np.inf] = 0

        occ_mask = Image.open(train_gt[i].replace('disp0GT.pfm', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32)

        mask = (disp_gt <= 0) | (occ_mask != 255) | (disp_gt >= args.maxdisp)
        # mask = (disp_gt <= 0) | (disp_gt >= maxdisp)

        error = np.abs(pred_np - disp_gt)
        error[mask] = 0
        
        print("#######Bad", limg_path, np.sum(error > 2.0) / (w * h - np.sum(mask)))

        op += np.sum(error > 2.0) / (w * h - np.sum(mask))
        mae += np.sum(error) / (w * h - np.sum(mask))

        ########save

        filename = os.path.join('./demo-imgs/', limg_path.split('/')[-2]+limg_path.split('/')[-1])
        pred_np_save = np.round(pred_np * 256).astype(np.uint16)        
        cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(pred_np_save, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    print("######Bad 2.0", op / 15 * 100)
    print("######EPE", mae / 15)


if __name__ == '__main__':
    test_trainset()
