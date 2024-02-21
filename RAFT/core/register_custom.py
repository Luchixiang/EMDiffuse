import sys
sys.path.append('core')
import argparse
import torch
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from align_functions import *
import torch.nn.functional as F
import os
from tifffile import imwrite
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import glob


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


def process_pair(wf_img, gt_img, save_wf_path, save_gt_path, patch_size=256, stride=224, model=None, border=32):
    wf_img = cv2.cvtColor(wf_img, cv2.COLOR_BGR2GRAY)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    (h, w) = gt_img.shape[:2]

    H = align_images(wf_img, gt_img)
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
                if np.sum(crop_wf_img[border:-border, border:-border] == 0) > 10:
                    print(f'warning, {save_wf_path}, {count}, {np.sum(crop_wf_img[border:-border, border:-border] == 0)}')
                    count += 1
                    y += stride
                    continue
            image1 = load_image(crop_gt_img)
            image2 = load_image(crop_wf_img)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            image_warped = warp(image2 / 255.0, flow_up)
            crop_wf_img = image_warped[0].permute(1, 2, 0).cpu().numpy()
            crop_wf_img = np.uint8(crop_wf_img[:, :, 0] * 255)
            if np.sum(crop_wf_img[border:-border, border:-border] == 0) > 10:
                print(f'after optical flow warning, {save_wf_path}, {count}, {np.sum(crop_wf_img[border:-border, border:-border] == 0)}')
                count += 1
                y += stride
                continue
            imwrite(os.path.join(save_wf_path, str(count) + '.tif'), crop_wf_img[border:-border, border:-border])
            imwrite(os.path.join(save_gt_path, str(count) + '.tif'), crop_gt_img[border:-border, border:-border])
            count += 1
            y += stride
        x += stride


def registration(args, img_list, gt_list, patch_size=256, overlap=0.125,border=32):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

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
        for img_count in range(len(img_list)):
            img = img_list[img_count]
            if not img.endswith('tif'):
                continue
            roi_wf_path = os.path.join(train_wf_path, str(img_count))
            roi_gt_path = os.path.join(train_gt_path, str(img_count))
            mkdir(roi_wf_path)
            mkdir(roi_gt_path)
            print(f'processing{img}')
            save_wf_path = os.path.join(roi_wf_path, 'noise1')
            save_gt_path = os.path.join(roi_gt_path, 'noise1')
            mkdir(save_wf_path)
            mkdir(save_gt_path)
            gt_file_img = cv2.imread(gt_list[img_count])
            wf_file_img = cv2.imread(img_list[img_count])
            process_pair(wf_file_img, gt_file_img, save_wf_path, save_gt_path, model=model,
                         patch_size=patch_size, stride=int(patch_size * (1-overlap)), border=border)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="../models/raft-things.pth")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--category', help="save warped images")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--occlusion', action='store_true', help='predict occlusion masks')
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--border', default=32, type=int)
    parser.add_argument('--overlap', default=0.125, type=float)

    args = parser.parse_args()
    img_list = []
    gt_list = []
    for file in os.listdir(os.path.join(args.path, 'img')):
        img_list.append(os.path.join(args.path, 'img', file))
        gt_list.append(os.path.join(args.path, 'gt',  file))
    registration(args, img_list, gt_list, patch_size=args.patch_size, overlap=args.overlap, border=args.border)
