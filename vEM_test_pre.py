import os
from tifffile import imread, imwrite
import numpy as np
import shutil
def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def prctile_norm(x, min_prc=0, max_prc=100):
    x = np.array(x).astype(np.float64)
    y = (x - min_prc) / (max_prc - min_prc + 1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y * 255.

def recon_pre(root_path):
    target_path = os.path.join(root_path, 'crop_patches')
    mkdir(os.path.join(target_path))
    for file in os.listdir(root_path):
        if 'tif' not in file:
            continue
        x = 0
        # index = file.split('_')[1][:-4]
        index = file[:-4]
        path_size = 256
        stride = 224
        img = imread(os.path.join(root_path, file))
        col_num = 0
        os.makedirs(os.path.join(target_path, index))
        while x + path_size <= img.shape[0]:
            y = 0
            row_num = 0
            x_start = x
            while y + path_size <= img.shape[1]:
                y_start = y
                patch = img[x_start:x_start + path_size, y_start:y_start + path_size]
                imwrite(os.path.join(target_path, index, str(row_num) + '_' + str(col_num) + '.tif'), patch)
                row_num += 1
                y += stride
            row_num += 1
            x += stride
            col_num += 1
    return os.path.join(root_path, 'crop_patches')

