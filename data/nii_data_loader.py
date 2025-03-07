import torch.utils.data as data

from PIL import Image
import os
import os.path

import sys
import SimpleITK as sitk
import numpy as np
import random

import cv2

IMG_EXTENSIONS = ['.nii.gz']


def is_nii_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()  # change upper case to lower case
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    photoClasses = [d for d in os.listdir(
        dir) if os.path.isfile(os.path.join(dir, d))]
    photoClasses.sort()
    photo_class_to_idx = {photoClasses[i]: i for i in range(len(photoClasses))}
    return photoClasses, photo_class_to_idx


def make_dataset(dir, photo_class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isfile(d):
            continue
        path = d
        item = (path, photo_class_to_idx[target])
        images.append(item)

    return images


def collect_nii_path(path):
    # walk all the .nii files in path
    all_file_list = []
    gci(path, all_file_list)
    all_file_list.append(path)

    return all_file_list


def gci(filepath, all_file_list):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            all_file_list.append(fi_d)
            gci(fi_d, all_file_list)


def nii_slides_loader(nii_file_path, num, transform=None, crop=None):
    """
    load nii slides, transform and crop it

    num: layer to slice
    transform: how to normalize
    crop: choose multiple boxes to crop for augmentation
    """
    item = sitk.ReadImage(nii_file_path)
    nii_slides = sitk.GetArrayFromImage(item)
    if len(nii_slides.shape) == 3:
        nii_slides = nii_slides[num, :, :]
    if transform is not None:
        nii_slides = transform(nii_slides)
    # nii_slides = nii_slides[num, :, :]
    if crop is not None:
        nii_slides = nii_slides[crop[0]:crop[1], crop[2]:crop[3]]
    return nii_slides


def matrix_resize(filein, sacle_size, crop_size, random_crop_para):
    # TODO:
    temp = np.reshape(filein, [sacle_size, sacle_size])


def normalize_nii(mrnp):
    matLPET = mrnp / mrnp.max() * 2.0 - 1
    return matLPET


def load_set(path):
    classes, class_to_idx = find_classes(path)
    loaded_set = make_dataset(path, class_to_idx)
    if len(loaded_set) == 0:
        raise (RuntimeError("Found 0 images in subfolders of: " + path + "\n"
            "Supported image extensions are: " + ",".join(
            IMG_EXTENSIONS)))
    return loaded_set

# normalize segmentation
def seg_transform(seg):
    seg = np.where(seg > 0, 1, np.finfo(float).eps)
    return seg

# find layer which has biggest tumor
def findSegBox(seg_path):

    item = sitk.ReadImage(seg_path)
    nii_slides = sitk.GetArrayFromImage(item)
    nii_slides[nii_slides>0] = 1

    maxID = 0
    aera = 0
    len_slides = len(nii_slides)

    # find the biggest tumor
    for i in range(len_slides):
        tmpAera = nii_slides[i, :, :].sum()
        if tmpAera == 0:
            continue
        elif tmpAera > aera:
            aera = tmpAera
            maxID = i

    return maxID


# find layers which has biggest and 1/2, 1/3, 1/4, 1/5 tumor
def findSegBox5(seg_path):

    item = sitk.ReadImage(seg_path)
    nii_slides = sitk.GetArrayFromImage(item)
    nii_slides[nii_slides>0] = 1
    c, h, w = nii_slides.shape

    maxID = 0
    aera = 0
    len_slides = len(nii_slides)
    res = []

    # find biggest tumor
    for i in range(len_slides):
        tmpAera = nii_slides[i, :, :].sum()
        if tmpAera == 0:
            continue
        elif tmpAera > aera:
            aera = tmpAera
            maxID = i
    res.append(maxID)

    maxRatio = aera / (h*w)

    # find the rest tumor, if absent, add biggest one until it's filled
    count = 0
    flag = [0,0,0,0]
    if maxRatio>0.02:
        for i in range(len_slides):
            tmpAera = nii_slides[i, :, :].sum()
            if tmpAera == 0:
                continue
            elif flag[0]==0 and tmpAera <= aera/2 and tmpAera > aera/3:
                count += 1
                flag[0]=1
                res.append(i)
            elif flag[1]==0 and tmpAera <= aera/3 and tmpAera > aera/4:
                count += 1
                flag[1]=1
                res.append(i)
            elif flag[2]==0 and tmpAera <= aera/4 and tmpAera > aera/5:
                count += 1
                flag[2]=1
                res.append(i)
            elif flag[3]==0 and tmpAera <= aera/5 and tmpAera > aera/6:
                count += 1
                flag[3]=1
                res.append(i)
            if count>=4:
                break
    else:
        for i in range(4):
            count += 1
            res.append(maxID)
    if count < 4:
        res += [maxID] * (4-count)
        count = 4

    return res

# load 2D image and normalize it
def img_loader(path, transform=None):
    item = Image.open(path).convert('L')
    nii_slides = item
    if transform is not None:
        nii_slides = transform(nii_slides)
    # nii_slides = nii_slides[num, :, :]
    return nii_slides

# normalize image with fixed 255.0
def normalize_img(mrnp):  # mrnp: original mri data
    # maxPercentPET, minPercentPET = np.percentile(mrnp, [99.5, 0])
    # # print('maxPercentPET: ', maxPercentPET, ' minPercentPET: ', minPercentPET)
    # matLPET = (mrnp - minPercentPET) / (maxPercentPET - minPercentPET)

    matLPET = np.array(mrnp)
    matLPET = matLPET / 255.0 * 2.0 - 1
    # matLPET = (matLPET - matLPET.mean()) / matLPET.std()

    return matLPET

# change seg to a matrix with 0 and 1
def seg_transform_img(seg):
    seg = np.array(seg)
    seg = np.where(seg > 0, 1, np.finfo(float).eps)
    return seg
