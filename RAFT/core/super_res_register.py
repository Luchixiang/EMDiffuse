import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from align_functions import *
import torch.nn.functional as F
import os
from tifffile import imwrite

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim)

    # return the resized image
    return resized


######################
## Image form trans###
######################
def img2tensor(img):
    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))

    return img_t


def tensor2img(img_t):
    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img


######################
# occlusion detection#
######################

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=False)
    return output



###########################
## raft functions
###########################
def load_image(img):
    img = np.stack([img, img, img], axis=2)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def process_pair(wf_img, gt_img, save_wf_path, save_gt_path, sup_wf_img=None, patch_size=256, stride=224, model=None,
                 border=32):
    wf_img_origin = cv2.cvtColor(wf_img, cv2.COLOR_BGR2GRAY)
    gt_img_origin = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    wf_img = wf_img_origin[gt_img_origin.shape[0] // 2 - gt_img_origin.shape[0] // 4:  gt_img_origin.shape[0] // 2 +
                                                                                       gt_img_origin.shape[0] // 4,
             gt_img_origin.shape[1] // 2 - gt_img_origin.shape[1] // 4: gt_img_origin.shape[1] // 2 +
                                                                        gt_img_origin.shape[1] // 4]
    gt_img = cv2.resize(gt_img_origin, (gt_img_origin.shape[1] // 2, gt_img_origin.shape[0] // 2))
    x_offset, y_offset, _, _ = image_registration.chi2_shift(wf_img, gt_img, 0.1, return_error=True)
    wf_img = shift(wf_img_origin, (y_offset, x_offset))[
             gt_img_origin.shape[0] // 2 - gt_img_origin.shape[0] // 4 - 8:  gt_img_origin.shape[0] // 2 +
                                                                             gt_img_origin.shape[0] // 4 + 8,
             gt_img_origin.shape[1] // 2 - gt_img_origin.shape[1] // 4 - 8: gt_img_origin.shape[1] // 2 +
                                                                            gt_img_origin.shape[1] // 4 + 8]

    H = align_images(wf_img, gt_img, debug=False)
    h, w = gt_img.shape
    aligned = cv2.warpPerspective(wf_img, H, (w, h))
    x = border
    x_end = wf_img.shape[0] - border
    y_end = wf_img.shape[0] - border
    count = 1
    while x + patch_size < x_end:
        y = border
        while y + patch_size < y_end:
            crop_wf_img = aligned[x - border: x + patch_size + border, y - border: y + patch_size + border]
            crop_gt_img = gt_img[x - border: x + patch_size + border, y - border: y + patch_size + border]
            H_sub = align_images(crop_wf_img, crop_gt_img)
            if H_sub is None:
                count += 1
                y += stride
                continue
            else:
                (h_sub, w_sub) = crop_gt_img.shape[:2]
                crop_wf_img = cv2.warpPerspective(crop_wf_img, H_sub, (w_sub, h_sub))
                if np.sum(crop_wf_img[border:-border, border:-border] == 0) > 10:
                    count += 1
                    y += stride
                    continue
            image1 = load_image(crop_gt_img)
            image2 = load_image(crop_wf_img)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            image_warped = warp(image2 / 255.0, flow_up)
            crop_wf_img = image_warped[0].permute(1, 2, 0).cpu().numpy()
            crop_wf_img = np.uint8(crop_wf_img[:, :, 0] * 255)
            if np.sum(crop_wf_img[border:-border, border:-border] == 0) > 10:
                count += 1
                y += stride
                continue
            imwrite(os.path.join(save_wf_path, str(count) + '.tif'), crop_wf_img[border:-border, border:-border])
            imwrite(os.path.join(save_gt_path, str(count) + '.tif'),
                    gt_img_origin[2 * x: 2 * x + 2 * patch_size, 2 * y: 2 * y + 2 * patch_size])
            count += 1
            y += stride
        x += stride


def registration(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        task = 'zoom'
        path = args.path
        target_path = os.path.join(path, task)
        mkdir(target_path)
        image_types = ['Brain__2w_01.tif', 'Brain__2w_02.tif', 'Brain__2w_03.tif']
        train_wf_path = os.path.join(target_path, 'train_wf')
        train_gt_path = os.path.join(target_path, 'train_gt')
        mkdir(train_wf_path)
        mkdir(train_gt_path)
        for i in range(100):
            if not os.path.exists(os.path.join(path, str(i), 'Brain__4w_09.tif')):
                continue
            roi_wf_path = os.path.join(train_wf_path, str(i))
            roi_gt_path = os.path.join(train_gt_path, str(i))
            
            mkdir(roi_wf_path)
            mkdir(roi_gt_path)
            for type in image_types:
                print(f'processing image {i}, {type}')
                save_wf_path = os.path.join(roi_wf_path, type[:-4])
                save_gt_path = os.path.join(roi_gt_path, type[:-4])
                mkdir(save_wf_path)
                mkdir(save_gt_path)
                gt_file_img = cv2.imread(os.path.join(path, str(i), 'Brain__4w_09.tif'))
                wf_file_img = cv2.imread(os.path.join(path, str(i), type))
                sup_wf_img = None
                # print(wf_file_img.min())
                process_pair(wf_file_img, gt_file_img, save_wf_path, save_gt_path, sup_wf_img=sup_wf_img, model=model,
                             patch_size=args.patch_size, border=args.border, stride=int(args.patch_size * (1-args.overlap)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="../models/raft-things.pth")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--category', help="save warped images")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--occlusion', action='store_true', help='predict occlusion masks')
    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--border', default=32, type=int)
    parser.add_argument('--overlap', default=0.125, type=float)

    args = parser.parse_args()

    registration(args)
