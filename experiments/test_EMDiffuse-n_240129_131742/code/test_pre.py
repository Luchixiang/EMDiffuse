import argparse
import os
import shutil

import cv2
from tifffile import imwrite


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def process_denoise_pair(wf_img, save_wf_path, path_size=256, stride=224):
    # print(wf_image.shape)
    if len(wf_img.shape) > 2:
        wf_img = cv2.cvtColor(wf_img, cv2.COLOR_BGR2GRAY)
    board = 0
    x = board
    x_end = wf_img.shape[0] - board
    y_end = wf_img.shape[0] - board
    row = 0
    while x + path_size < x_end:
        y = board
        col = 0
        while y + path_size < y_end:
            crop_wf_img = wf_img[x: x + path_size, y : y + path_size]

            imwrite(os.path.join(save_wf_path, str(row) + '_' + str(col) + '.tif'),
                    crop_wf_img)
            col += 1
            y += stride
        row += 1
        x += stride


def test_pre(data_root, task='denoise'):
    target_path = os.path.join(data_root, task + '_test_crop_patches')
    mkdir(target_path)
    if task == 'denoise':
        image_types = ['Brain__4w_04.tif', 'Brain__4w_05.tif', 'Brain__4w_06.tif', 'Brain__4w_07.tif',
                       'Brain__4w_08.tif']
    else:
        image_types = ['Brain__2w_01.tif', 'Brain__2w_02.tif', 'Brain__2w_03.tif']
    for region_index in os.listdir(data_root):
        if not region_index.isdigit():
            continue
        mkdir(os.path.join(target_path, region_index))
        for type in image_types:
            # mkdir(os.path.join(target_path, region_index, type))
            save_wf_path = os.path.join(os.path.join(target_path, region_index, type[:-4]))
            mkdir(save_wf_path)
            print(os.path.join(data_root, region_index, type))
            wf_file_img = cv2.imread(os.path.join(data_root, region_index, type))
            if task == 'denoise':
                process_denoise_pair(wf_file_img, save_wf_path, path_size=256, stride=224)
            else:
                process_denoise_pair(wf_file_img, save_wf_path, path_size=128, stride=112)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="denoise")
    parser.add_argument('--path', help="dataset for evaluation")
    args = parser.parse_args()
    test_pre(args.path, args.task)