import numpy as np
import os
from tifffile import imread, imwrite
import argparse


def find_max_number(folder_path):
    max_number = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            filename = filename[:-4]
        if not filename.isdigit():
            continue
        filename = int(filename)

        number = int(filename)
        max_number = max(max_number, number)
    return max_number


def mkdir(path):
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    os.mkdir(path)


def vem_transpose(data_root):
    stacks = []
    z_depth = find_max_number(data_root)

    for i in range(z_depth):
        stacks.append(imread(os.path.join(data_root, f'{i}.tif')))
    stack = np.stack(stacks)
    print(stack.shape)
    stack = stack.transpose(1, 0, 2)
    target_file_path = os.path.join(data_root, 'transposed')
    mkdir(target_file_path)
    for i in range(stack.shape[0]):
        imwrite(os.path.join(target_file_path, str(i) + '.tif'), stack[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task', default="denoise")
    parser.add_argument('--path', help="dataset for evaluation")
    args = parser.parse_args()
    vem_transpose(args.path)
