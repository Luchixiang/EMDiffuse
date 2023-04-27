import os
from tifffile import imread, imwrite
import shutil
import cv2
import imutils
import numpy as np
import image_registration
from scipy.ndimage import shift
import os
import cv2
import pyiqa
import torch


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def delete_outlier(points1, points2, move=0, outlier_percent=0.3):
    """
    Delete the outliers/mismatches based on the angle and distance.
    Args:
        points1: key points detected in frame1
        points2: key points detected in frame2
        move: move for a small distance to avoid points appear at the same location
        outlier_percent: how many outliers are removed

    Returns: indexes of selected key points
    """
    # angle
    points1_mv = points1.copy()
    points1_mv[:, 0] = points1[:, 0] - move
    vecs = points2 - points1_mv
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vec_norms = vecs / (norms + 1e-6)
    vec_means = np.mean(vec_norms, axis=0).reshape((2, 1))
    cross_angles = vec_norms.dot(vec_means)[:, 0]
    index = np.argsort(-cross_angles)
    num_select = int(len(index) * (1 - outlier_percent))
    index_selected = index[0:num_select]

    # distance
    index1 = np.argsort(norms[:, 0])
    # print(index1)
    index1_selected = index1[0:num_select]

    index_selected = list(set(index1_selected) & set(index_selected))

    return index_selected, np.mean(norms[index_selected])


def align_images(imageGray, templateGray, maxFeatures=500, keepPercent=0.2,
                 debug=False, outlier=True, sup_img=None):
    # convert both the input image and template to grayscale

    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt
    if outlier:
        index, distance = delete_outlier(ptsA, ptsB)

        matches = list(np.array(matches)[index])
        ptsA = ptsA[index, :]
        ptsB = ptsB[index, :]
    if len(matches) < 10:
        print('can not match')
        return None
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    return H


def process_denoise_pair(wf_img, save_wf_path, path_size=256, stride=224):
    # print(wf_image.shape)
    device = torch.device('cpu')
    metirc = pyiqa.create_metric('brisque').to(device)
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


def estimate_noise_level(root_path):
    print('estimating noise level')
    device = torch.device('cpu')
    metirc = pyiqa.create_metric('brisque').to(device)
    noise_dict = dict()
    for file in sorted(os.listdir(root_path)):
        if file.endswith('tif'):
            print(file)
            path = os.path.join(root_path, file)
            a = cv2.imread(path)
            if len(a.shape) == 3:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

            a = torch.tensor(a).unsqueeze_(dim=0).unsqueeze_(dim=0).float()
            # a = a / a.max()
            a = a / 255.
            # a = (a - a.min()) / (a.max() - a.min())
            noise = metirc(a)
            noise_dict[file] = noise
    return noise_dict


def dynamic_mean(noise_level, task='denoise'):
    if task != 'denoise':
        return 2
    if noise_level < 55:
        return 1
    elif noise_level < 78:
        return 2
    else:
        return 4


def denoise_pre(data_root, phase='val', task='denoise'):
    target_path = os.path.join(data_root, 'crop_patches')
    if os.path.exists(target_path):
        return os.path.join(data_root, 'crop_patches')
    mkdir(target_path)
    noise_dict = estimate_noise_level(data_root)
    print(noise_dict)
    for file in os.listdir(data_root):
        # print(file)
        if 'tif' not in file:
            continue
        print(f'pre-processing{file}')
        save_wf_path = os.path.join(os.path.join(target_path, file[:-4] + '_' + str(dynamic_mean(noise_dict[file], task))))
        mkdir(save_wf_path)
        wf_file_img = cv2.imread(os.path.join(data_root, file))
        if task == 'denoise':
            process_denoise_pair(wf_file_img, save_wf_path, path_size=256, stride=224)
        else:
            process_denoise_pair(wf_file_img, save_wf_path, path_size=128, stride=112)
    return os.path.join(data_root, 'crop_patches')
