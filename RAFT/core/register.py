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

DEVICE = 'cpu'


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
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output


def compute_flow_magnitude(flow):
    flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2

    return flow_mag


def compute_flow_gradients(flow):
    H = flow.shape[0]
    W = flow.shape[1]

    flow_x_du = np.zeros((H, W))
    flow_x_dv = np.zeros((H, W))
    flow_y_du = np.zeros((H, W))
    flow_y_dv = np.zeros((H, W))

    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv


def detect_occlusion(fw_flow, bw_flow):
    ## fw-flow: img1 => img2, tensor N,C,H,W
    ## bw-flow: img2 => img1, tensor N,C,H,W

    with torch.no_grad():
        ## warp fw-flow to img2
        fw_flow_w = warp(fw_flow, bw_flow)

        ## convert to numpy array
        fw_flow_w = tensor2img(fw_flow_w)
        fw_flow = tensor2img(fw_flow)
        bw_flow = tensor2img(bw_flow)

    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2

    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = np.logical_or(mask1, mask2)
    occlusion = np.zeros((fw_flow.shape[0], fw_flow.shape[1]))
    occlusion[mask == 1] = 1

    return occlusion


###########################
## raft functions
###########################
def load_image(img):
    img = np.stack([img, img, img], axis=2)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    import matplotlib.pyplot as plt
    plt.imshow(img_flo / 255.0)
    plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


def process_pair(wf_img, gt_img, save_wf_path, save_gt_path, sup_wf_img=None, patch_size=256, stride=224, model=None):
    wf_img = cv2.cvtColor(wf_img, cv2.COLOR_BGR2GRAY)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    (h, w) = gt_img.shape[:2]
    if sup_wf_img is not None:
        sup_wf_img = cv2.cvtColor(sup_wf_img, cv2.COLOR_BGR2GRAY)
        x_offset, y_offset, _, _ = image_registration.chi2_shift(wf_img, sup_wf_img, 0.1, return_error=True)
        print(y_offset, x_offset)
        wf_img = shift(wf_img, (y_offset, x_offset))
        H = align_images(sup_wf_img, gt_img)
        sub_aligned = cv2.warpPerspective(sup_wf_img, H, (w, h))
    else:
        H = align_images(wf_img, gt_img)
    aligned = cv2.warpPerspective(wf_img, H, (w, h))
    board = 32
    x = board
    x_end = wf_img.shape[0] - board
    y_end = wf_img.shape[0] - board
    count = 1
    while x + patch_size < x_end:
        y = board
        while y + patch_size < y_end:
            crop_wf_img = aligned[x - board: x + patch_size + board, y - board: y + patch_size + board]
            crop_gt_img = gt_img[x - board: x + patch_size + board, y - board: y + patch_size + board]
            if sup_wf_img is not None:
                crop_sup_wf_img = sub_aligned[x - board: x + patch_size + board, y - board: y + patch_size + board]
                H_sub = align_images(crop_sup_wf_img, crop_gt_img)
            else:
                try:
                    H_sub = align_images(crop_wf_img, crop_gt_img)
                except:
                    print(f'can not align{count}, {save_wf_path}')
                    count += 1
                    y += stride
                    continue

            if H_sub is None:
                print(f'can not align{count}, {save_wf_path}')
            else:
                (h_sub, w_sub) = crop_gt_img.shape[:2]
                crop_wf_img = cv2.warpPerspective(crop_wf_img, H_sub, (w_sub, h_sub))
                if np.sum(crop_wf_img[board:-board, board:-board] == 0) > 10:
                    print(f'warning, {save_wf_path}, {count}, {np.sum(crop_wf_img[board:-board, board:-board] == 0)}')
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
            imwrite(os.path.join(save_wf_path, str(count) + '.tif'), crop_wf_img[board:-board, board:-board])
            imwrite(os.path.join(save_gt_path, str(count) + '.tif'), crop_gt_img[board:-board, board:-board])
            count += 1
            y += stride
        x += stride


def registration(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    tissue = args.tissue
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        task = 'denoise'
        path = args.path
        target_path = os.path.join(path, task)
        mkdir(target_path)

        train_wf_path = os.path.join(target_path, 'train_wf')
        train_gt_path = os.path.join(target_path, 'train_gt')
        mkdir(train_wf_path)
        mkdir(train_gt_path)
        for i in range(2, 100):
            if not os.path.exists(os.path.join(path, str(i), tissue +'__4w_09.tif')):
                continue
            roi_wf_path = os.path.join(train_wf_path, str(i))
            roi_gt_path = os.path.join(train_gt_path, str(i))
            mkdir(roi_wf_path)
            mkdir(roi_gt_path)
            for type in os.listdir(os.path.join(path, str(i))):
                if '_09' in type or '2w' in type:
                    continue
                print(f'processing{i}, {type}')
                save_wf_path = os.path.join(roi_wf_path, type[:-4])
                save_gt_path = os.path.join(roi_gt_path, type[:-4])
                mkdir(save_wf_path)
                mkdir(save_gt_path)
                gt_file_img = cv2.imread(os.path.join(path, str(i), tissue +'__4w_09.tif'))
                wf_file_img = cv2.imread(os.path.join(path, str(i), type))
                sup_wf_img = None
                if '_04' in type or '05' in type and tissue == 'brain':
                    sup_wf_img = cv2.imread(os.path.join(path, str(i), tissue +'__4w_06.tif'))
                # print(wf_file_img.min())
                process_pair(wf_file_img, gt_file_img, save_wf_path, save_gt_path, sup_wf_img=sup_wf_img, model=model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="../models/raft-things.pth")
    parser.add_argument('--tissue', default='Brain')
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--category', help="save warped images")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--occlusion', action='store_true', help='predict occlusion masks')

    args = parser.parse_args()

    registration(args)