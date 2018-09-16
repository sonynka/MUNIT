"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
from logging import getLogger
import pandas as pd
import random

logger = getLogger()

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, mode='train'):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader
        self.mode = mode

        num_train_set = int(0.9 * len(self.imlist))
        self.train_imgs = self.imlist[:num_train_set]
        self.test_imgs = self.imlist[num_train_set:]

        if self.mode == 'train':
            self.num_data = len(self.train_imgs)
        elif self.mode == 'test':
            self.num_data = len(self.test_imgs)

    def __getitem__(self, index):

        if self.mode == 'train':
            impath = self.train_imgs[index]
        elif self.mode == 'test':
            impath = self.test_imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.num_data


class UnalignedDataset(data.Dataset):

    def __init__(self, root, flist_path, labels_path, attrA, attrB,
                 categories=None, transform=None, flist_loader=default_flist_reader,
                 loader=default_loader):

        self.root = root
        self.loader = loader
        self.labels_path = labels_path
        self.transform = transform

        imlist = flist_loader(flist_path)
        if categories is not None:
            catlist = categories.split(',')
            imlist = [im for im in imlist if any(cat in im for cat in catlist)]

        self.imlistA, self.imlistB = self.process_labels(attrA, attrB, imlist)

        self.A_size = len(self.imlistA)
        self.B_size = len(self.imlistB)

    def process_labels(self, attrA, attrB, imlist):
        """ Load the labels from CSV file, process, and return a dataframe """

        labels_df = pd.read_csv(self.labels_path)
        labels_df = labels_df.loc[labels_df.img_path.isin(imlist)]

        imlistA = labels_df.loc[labels_df[attrA] == 1].img_path.tolist()
        if attrB is None:
            imlistB = labels_df.loc[labels_df[attrA] == 0].img_path.tolist()
        else:
            imlistB = labels_df.loc[labels_df[attrB] == 1].img_path.tolist()

        return imlistA, imlistB

    def __getitem__(self, index):

        pathA = os.path.join(self.root, self.imlistA[index])
        pathB_idx = random.randint(0, self.B_size - 1)
        pathB = os.path.join(self.root, self.imlistB[pathB_idx])

        imgA = self.loader(pathA)
        imgB = self.loader(pathB)

        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        return {'A': imgA, 'B': imgB,
                'A_paths': pathA, 'B_paths': pathB}

    def __len__(self):
        return min(self.A_size, self.B_size)

class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader, mode='train'):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.mode = mode

        num_train_set = int(0.9 * len(imgs))
        self.train_imgs = imgs[:num_train_set]
        self.test_imgs = imgs[num_train_set:]

        if self.mode == 'train':
            self.num_data = len(self.train_imgs)
        elif self.mode == 'test':
            self.num_data = len(self.test_imgs)

    def __getitem__(self, index):

        path = ''
        if self.mode == 'train':
            path = self.train_imgs[index]
        elif self.mode == 'test':
            path = self.test_imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return self.num_data
