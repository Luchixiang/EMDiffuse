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

import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
def load_image(imfile):
    if type(imfile) == str:
        img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    else:
        img = imfile
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


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        # data_root = '/userhome/34/cxlu/c2f/point'
        data_root = '/mnt/sdc/cxlu/c2f/point'
        for home, dirs, files in os.walk(data_root):
            for filename in files:
                if filename == 'Fused.tif':
                    region_file = os.path.join(home, 'regions.json')
                    print(region_file)
                    f = open(region_file)

                    # returns JSON object as
                    # a dictionary
                    import json
                    data = json.load(f)
                    coarse_img = data['coarse']
                    coarse_img = os.path.join(home, coarse_img)
                    fused_img = os.path.join(home, filename)
                    coarse_img = cv2.imread(coarse_img)
                    fused_img = cv2.imread(fused_img)
                    startx = data['startx']
                    starty = data['starty']
                    endx = data['endx']
                    endy = data['endy']
                    coarse_img = coarse_img[starty:endy, startx:endx, :]
                    h, w = coarse_img.shape[:2]
                    if h < w:
                        coarse_img = image_resize(coarse_img, None, 512)
                    else:
                        coarse_img = image_resize(coarse_img, 512, None)
                    fused_img = cv2.resize(fused_img, coarse_img.shape[:2][::-1])
                    image1 = load_image(fused_img)
                    image2 = load_image(coarse_img)

                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

                    image_warped = warp(image2 / 255.0, flow_up)
                    img = image_warped[0].permute(1, 2, 0).cpu().numpy()
                    cv2.imwrite(os.path.join(home, 'aligned_lr_images_optical.tif'), np.uint8(img[:, :, ::-1] * 255))
                    import matplotlib
                    import matplotlib.pyplot as plt
                    matplotlib.use('Agg')
                    fig, ax = plt.subplots(2, 2)
                    image1 = image1[0].permute(1, 2, 0).cpu().numpy()
                    image2 = image2[0].permute(1, 2, 0).cpu().numpy()
                    ax[0, 0].set_title('img_src')
                    ax[0, 0].imshow(cv2.cvtColor(np.uint8(image1[:, :, ::-1] * 255), cv2.COLOR_BGR2RGB))
                    ax[0, 1].set_title('img_templ')
                    ax[0, 1].imshow(cv2.cvtColor(np.uint8(image2[:, :, ::-1] * 255), cv2.COLOR_BGR2RGB))
                    ax[1, 0].set_title('result')
                    ax[1, 0].imshow(cv2.cvtColor(np.uint8(img[:, :, ::-1] * 255), cv2.COLOR_BGR2RGB))
            # ax[1, 1].set_title('matches')
            # ax[1, 1].imshow(cv2.cvtColor(imMatch, cv2.COLOR_BGR2RGB))
            # paths = home.split('/')[3:]
                    name = os.path.join('./registered', home.replace('/', '_') + '.png')
                    print(name)
                    plt.savefig(name)
            # cv2.imwrite(imfile3, np.uint8(img[:, :, ::-1] * 255))
            # print(imfile3)
        # categories = ['Reds', 'sports', 'text', 'daily', 'animal', 'Moca', 'house', 'landscape']
        # for k in range(len(categories)):
        #     category = categories[k]
        #
        #     ### save warped image
        #     for j in range(300):
        #         clean_frame_root = args.path + 'processed/frames/val/%s/video_%s' % (category, j)
        #         step1_frame_root = args.path + 'homo_align/iphone/frames/val/%s/video_%s' % (category, j)
        #         step2_frame_root = args.path + 'homo_flow_align/iphone/frames/val/%s/video_%s' % (category, j)
        #         if not (os.path.isdir(clean_frame_root) and os.path.isdir(step1_frame_root)):
        #             continue
        #         if not os.path.isdir(step2_frame_root):
        #             os.makedirs(step2_frame_root)
        #
        #         for i in range(60):
        #             imfile1 = clean_frame_root + '/' + '%05d.jpg' % (i + 15)
        #             imfile2 = step1_frame_root + '/' + '%05d.jpg' % (i)
        #             imfile3 = step2_frame_root + '/' + '%05d.jpg' % (i)
        #             image1 = load_image(imfile1)
        #             image2 = load_image(imfile2)
        #
        #             padder = InputPadder(image1.shape)
        #             image1, image2 = padder.pad(image1, image2)
        #
        #             flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        #
        #             image_warped = warp(image2 / 255.0, flow_up)
        #             img = image_warped[0].permute(1, 2, 0).cpu().numpy()
        #             cv2.imwrite(imfile3, np.uint8(img[:, :, ::-1] * 255))
        #             print(imfile3)
        #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--category', help="save warped images")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--occlusion', action='store_true', help='predict occlusion masks')

    args = parser.parse_args()

    demo(args)
