import os
from tifffile import imread, imwrite
import shutil
import cv2
import imutils
import numpy as np
import image_registration
from scipy.ndimage import shift


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
        return None
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    return H



