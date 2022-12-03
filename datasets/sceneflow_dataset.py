import os
import random
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

class SceneFlowDatset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        #print('####path',self.left_filenames,self.right_filenames)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        # print(os.path.join(self.datapath, self.left_filenames[index]))
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        left_img_np = np.array(left_img)
        # print(left_img_np.shape)
        # plt.subplot(121)
        # plt.imshow(disparity_front)
        # plt.show()
        dx_imgL = cv2.Sobel(left_img_np,cv2.CV_32F,1,0,ksize=3)
        dy_imgL = cv2.Sobel(left_img_np,cv2.CV_32F,0,1,ksize=3)
        dxy_imgL=np.sqrt(np.sum(np.square(dx_imgL),axis=-1)+np.sum(np.square(dy_imgL),axis=-1))
        dxy_imgL = dxy_imgL/(np.max(dxy_imgL)+1e-5)
        #print('####11111', dxy_imgL.shape)
        # mask_edge = (dxy_imgL > np.percentile(dxy_imgL,80))
        # print(mask_img.shape, disparity.shape)

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            disparity_low = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)
            disparity_low_r8 = cv2.resize(disparity, (crop_w//8, crop_h//8), interpolation=cv2.INTER_NEAREST)
            # mask_edge = mask_edge[y1:y1 + crop_h, x1:x1 + crop_w]
            # mask_smooth = ~mask_edge
            gradient_map = dxy_imgL[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            # mask_edge = torch.from_numpy(np.array(mask_edge, dtype=np.uint8))
            # mask_smooth = torch.from_numpy(np.array(mask_smooth, dtype=np.uint8))
            gradient_map = torch.from_numpy(gradient_map)
            #print('###2222', gradient_map.shape, gradient_map)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "gradient_map":gradient_map,
                    "disparity_low":disparity_low,
                    "disparity_low_r8":disparity_low_r8}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]
            gradient_map = dxy_imgL[h - crop_h:h, w - crop_w: w]
            disparity_low = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)
            # mask_edge = mask_edge[h - crop_h:h, w - crop_w: w]
            # mask_smooth = ~mask_edge

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            # mask_edge = torch.from_numpy(np.array(mask_edge, dtype=np.uint8))
            # mask_smooth = torch.from_numpy(np.array(mask_smooth, dtype=np.uint8))

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0,
                    "gradient_map":gradient_map,
                    "disparity_low":disparity_low,
                    "left_filename": self.left_filenames[index]}
