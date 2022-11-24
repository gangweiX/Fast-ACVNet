import os, torch, torch.utils.data as data
from PIL import Image
import numpy as np
from . import flow_transforms
import pdb
import torchvision
import warnings
from . import readpfm as rp
from datasets.data_io import get_transform, read_all_lines
import cv2
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

IMG_EXTENSIONS = [
 '.jpg', '.JPG', '.jpeg', '.JPEG',
 '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any((filename.endswith(extension) for extension in IMG_EXTENSIONS))


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        return data
    else:
        data = rp.readPFM(path)[0]
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data


class myImageFloder(data.Dataset):

    def __init__(self, left, right, left_disparity, training, right_disparity=None, loader=default_loader, dploader=disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.disp_R = right_disparity
        self.training = training
        self.loader = loader
        self.dploader = dploader

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        left_img = self.loader(left)
        right_img = self.loader(right)
        if self.disp_L is not None:
          disp_L = self.disp_L[index]
          disparity = self.dploader(disp_L)
          disparity[disparity == np.inf] = 0
        else:
          disparity = None

        if self.training:
            th, tw = 256, 512
            #th, tw = 320, 704
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
            right_img = np.array(right_img)
            left_img = np.array(left_img)

            # w, h  = left_img.size
            # th, tw = 256, 512
            #
            # x1 = random.randint(0, w - tw)
            # y1 = random.randint(0, h - th)
            #
            # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            # dataL = dataL[y1:y1 + th, x1:x1 + tw]
            # right_img = np.asarray(right_img)
            # left_img = np.asarray(left_img)

            # geometric unsymmetric-augmentation
            angle = 0;
            px = 0
            if np.random.binomial(1, 0.5):
                # angle = 0.1;
                # px = 2
                angle = 0.05
                px = 1
            co_transform = flow_transforms.Compose([
                # flow_transforms.RandomVdisp(angle, px),
                # flow_transforms.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
                flow_transforms.RandomCrop((th, tw)),
            ])
            augmented, disparity = co_transform([left_img, right_img], disparity)
            left_img = augmented[0]
            right_img = augmented[1]

            right_img.flags.writeable = True
            if np.random.binomial(1,0.2):
              sx = int(np.random.uniform(35,100))
              sy = int(np.random.uniform(25,75))
              cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # to tensor, normalize
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            # print("disparity", disparity.shape)
            disparity_low = cv2.resize(disparity, (tw//4, th//4), interpolation=cv2.INTER_NEAREST)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    'disparity_low': disparity_low}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            if h % 64 == 0:
                top_pad = 0
            else:
                top_pad = 64 - (h % 64)

            if w % 64 == 0:
                right_pad = 0
            else:
                right_pad = 64 - (w % 64)
            assert top_pad >= 0 and right_pad >= 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left[index],
                        "right_filename": self.right[index]}

    def __len__(self):
        return len(self.left)
