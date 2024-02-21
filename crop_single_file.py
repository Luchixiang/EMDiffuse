import argparse
import os
import shutil
import numpy as np
import cv2
from tifffile import imwrite

def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def crop(wf_img, save_wf_path, patch_size=256, overlap=0.125):
    if len(wf_img.shape) > 2:
        wf_img = cv2.cvtColor(wf_img, cv2.COLOR_BGR2GRAY)
    if wf_img.dtype == np.uint16: # convert to 8 bit
        cmin = wf_img.min()
        cmax = wf_img.max()
        cscale = cmax - cmin
        if cscale == 0:
            cscale = 1
        scale = float(255 - 0) / cscale
        wf_img = (wf_img - cmin) * scale + 0
        wf_img = np.clip(wf_img, 0, 255) + 0.5
        wf_img = wf_img.astype(np.uint8)
    stride = int(patch_size * (1 - overlap))
    if len(wf_img.shape) > 2:
        wf_img = cv2.cvtColor(wf_img, cv2.COLOR_BGR2GRAY)
    border = 0
    
    x = border
    x_end = wf_img.shape[0] - border
    y_end = wf_img.shape[0] - border
    row = 0
    while x + patch_size < x_end:
        y = border
        col = 0
        while y + patch_size < y_end:
            crop_wf_img = wf_img[x: x + patch_size, y: y + patch_size]
            imwrite(os.path.join(save_wf_path, str(row) + '_' + str(col) + '.tif'),
                    crop_wf_img)
            col += 1
            y += stride
        row += 1
        x += stride


def test_pre(data_root, task='denoise'):
    target_path = os.path.join(data_root, task + '_test_crop_patches')
    mkdir(target_path)
    for file in os.listdir(data_root):
        if not file.endswith('tif'):
            continue
        mkdir(os.path.join(target_path, file[:-4]))
        save_wf_path = os.path.join(os.path.join(target_path, file[:-4], '0'))
        mkdir(save_wf_path)
        wf_file_img = cv2.imread(os.path.join(data_root, file))
        if task == 'denoise':
            crop(wf_file_img, save_wf_path, patch_size=256, overlap=0.125)
        else:
            crop(wf_file_img, save_wf_path, patch_size=128, overlap=0.125)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="denoise")
    parser.add_argument('--path', help="dataset for evaluation")
    args = parser.parse_args()
    test_pre(args.path, args.task)
