import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import os
import numpy as np
import torchvision.transforms.functional as TF
import random
import torch
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from tifffile import imread
from scipy.ndimage import zoom
class KidneySRUData(data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

        self.volume_list = self.read_dataset(data_root=self.data_root)

    def __getitem__(self, index):
        ret = {}
        gt = imread(self.volume_list[index])
        # print(gt.shape)
        img = gt[::6, :, :]
        img_upsampled = zoom(img, (6, 1,1 ), order=3)
        img, gt, img_upsampled = self.aug(img, gt, img_upsampled)
        img = img / 255.
        gt = gt / 255.
        img_upsampled = img_upsampled / 255.
        img = self.norm(img)
        gt = self.norm(gt)
        img_upsampled = self.norm(img_upsampled)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze_(dim=0)
        gt = torch.tensor(gt, dtype=torch.float32).unsqueeze_(dim=0)
        img_upsampled = torch.tensor(img_upsampled, dtype=torch.float32).unsqueeze_(dim=0)
        return img, gt,img_upsampled

    def norm(self, img):
        img = img.astype(np.float32)
        img = (img - 0.5) / 0.5
        return img

    def __len__(self):
        return len(self.volume_list)

    def aug(self, img, gt, img_up):
        if random.random() < 0.5:
            img = np.flip(img, axis=2)
            gt = np.flip(gt, axis=2)
            img_up = np.flip(img_up, axis=2)
        if random.random() < 0.5:
            img = np.rot90(img, k=1, axes=(1, 2))
            gt = np.rot90(gt, k=1, axes=(1, 2))
            img_up = np.rot90(img_up, k=1, axes=(1, 2))
        return img, gt, img_up

    def read_dataset(self, data_root):
        volume_list = []
        for i in range(2000):
            if os.path.exists(os.path.join(data_root, str(i) + '.tif')):
                volume_list.append(os.path.join(data_root, str(i) + '.tif'))
        return volume_list
