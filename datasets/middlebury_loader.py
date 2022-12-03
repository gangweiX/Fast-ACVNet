import os
from PIL import Image
from datasets import readpfm as rp
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random


def mb_loader(filepath, res):

    train_path = os.path.join(filepath, 'training' + res)
    test_path = os.path.join(filepath, 'test' + res)
    # gt_path = train_path.replace('training' + res, 'Eval3_GT/training' + res)
    gt_path = os.path.join(filepath, 'training' + res)

    train_left = []
    train_right = []
    train_gt = []

    for c in os.listdir(train_path):
        train_left.append(os.path.join(train_path, c, 'im0.png'))
        train_right.append(os.path.join(train_path, c, 'im1.png'))
        train_gt.append(os.path.join(gt_path, c, 'disp0GT.pfm'))

    test_left = []
    test_right = []
    for c in os.listdir(test_path):
        test_left.append(os.path.join(test_path, c, 'im0.png'))
        test_right.append(os.path.join(test_path, c, 'im1.png'))

    train_left = sorted(train_left)
    train_right = sorted(train_right)
    train_gt = sorted(train_gt)
    test_left = sorted(test_left)
    test_right = sorted(test_right)

    return train_left, train_right, train_gt, test_left, test_right


def img_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class myDataset(data.Dataset):

    def __init__(self, left, right, left_disp, training, imgloader=img_loader, dploader = disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disp
        self.imgloader = imgloader
        self.dploader = dploader
        self.training = training
        self.img_transorm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.imgloader(left)
        right_img = self.imgloader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = Image.fromarray(np.ascontiguousarray(dataL, dtype=np.float32))

        if self.training:
            w, h = left_img.size

            # random resize
            s = np.random.uniform(0.95, 1.05, 1)
            rw, rh = np.round(w*s), np.round(h*s)
            left_img = left_img.resize((rw, rh), Image.NEAREST)
            right_img = right_img.resize((rw, rh), Image.NEAREST)
            dataL = dataL.resize((rw, rh), Image.NEAREST)
            dataL = Image.fromarray(np.array(dataL) * s)

            # random horizontal flip
            p = np.random.rand(1)
            if p >= 0.5:
                left_img = horizontal_flip(left_img)
                right_img = horizontal_flip(right_img)
                dataL = horizontal_flip(dataL)

            w, h = left_img.size
            tw, th = 320, 240
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1+tw, y1+th))
            right_img = right_img.crop((x1, y1, x1+tw, y1+th))
            dataL = dataL.crop((x1, y1, x1+tw, y1+th))

            left_img = self.img_transorm(left_img)
            right_img = self.img_transorm(right_img)

            dataL = np.array(dataL)
            return left_img, right_img, dataL

        else:
            w, h = left_img.size
            left_img = left_img.resize((w // 32 * 32, h // 32 * 32))
            right_img = right_img.resize((w // 32 * 32, h // 32 * 32))

            left_img = self.img_transorm(left_img)
            right_img = self.img_transorm(right_img)

            dataL = np.array(dataL)
            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)


def horizontal_flip(img):
    img_np = np.array(img)
    img_np = np.flip(img_np, axis=1)
    img = Image.fromarray(img_np)
    return img


if __name__ == '__main__':
    train_left, train_right, train_gt, _, _ = mb_loader('/media/data/dataset/MiddEval3-data-Q/', res='Q')
    H, W = 0, 0
    for l in train_right:
        left_img = Image.open(l).convert('RGB')
        h, w = left_img.size
        H += h
        W += w
    print(H / 15, W / 15)