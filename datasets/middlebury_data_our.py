import os
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
from torchvision import transforms
import cv2
import torch
import torch.utils.data as data
import random

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def readlines(filename):
    """ read lines of a text file """
    with open(filename, 'r') as file_handler:
        lines = file_handler.read().splitlines()
    return lines

def read_pfm(file):
    with open(file, 'rb') as fh:
        fh.readline()
        width, height = str(fh.readline().rstrip())[2:-1].split()
        fh.readline()
        disp = np.fromfile(fh, '<f')
        return np.flipud(disp.reshape(int(height), int(width)))


class MiddleburyStereoDataset:

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 is_train=False,
                 has_gt=True,
                 **kwargs):

        self.img_resizer = transforms.Resize(size=(height, width))
        self.height = height
        self.width = width
        self.loader = pil_loader
        self.read_pfm = read_pfm
        self.data_path = data_path
        self.filenames = readlines(filenames)
        self.to_tensor = transforms.ToTensor()
        self.is_train = is_train
        self.has_gt = has_gt
        self.training = is_train

    def load_images(self, idx, do_flip=False):

        folder = self.filenames[idx]
        left_img = self.loader(os.path.join(self.data_path, 'MiddEval3/trainingF',
                                         folder, 'im0.png'))
        right_img = self.loader(os.path.join(self.data_path, 'MiddEval3/trainingF',
                                                folder, 'im1.png'))
        return left_img, right_img

    def load_disparity(self, idx, do_flip=False):
        folder = self.filenames[idx]
        disparity = self.read_pfm(os.path.join(self.data_path, 'MiddEval3/trainingF',
                                               folder, 'disp0GT.pfm'))
        # loaded disparity contains infs for no reading
        disparity[disparity == np.inf] = 0
        return np.ascontiguousarray(disparity)

    def preprocess(self, inputs):
        # convert to tensors and standardise using ImageNet
        for key in ['left_img', 'right_img']:
            inputs[key] = (self.to_tensor(inputs[key]) - 0.45) / 0.225

        if self.has_gt:
            inputs['disparity'] = torch.from_numpy(inputs['disparity']).float()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        inputs = {}
        left_img, right_img = self.load_images(idx, do_flip=False)
        disparity = self.load_disparity(idx)
        if self.is_train:
            w, h = left_img.size
            crop_w, crop_h = self.width, self.height
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            inputs['left_img'] = left_img
            inputs['right_img'] = right_img
            inputs['disparity'] = disparity
            self.preprocess(inputs)
            left_img = inputs['left_img']
            right_img = inputs['right_img']
            disparity = inputs['disparity']

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size
            print(w,h)
            crop_w, crop_h = self.width, self.height
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            inputs['left_img'] = left_img
            inputs['right_img'] = right_img
            inputs['disparity'] = disparity
            self.preprocess(inputs)
            left_img = inputs['left_img']
            right_img = inputs['right_img']
            disparity = inputs['disparity']

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}