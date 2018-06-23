#!/usr/bin/env python
import os
import matplotlib.image as mpimg
import numpy as np
import torch
import pandas as pd


N_IMAGES = 39788
IMG_SIZE = 256
IMG_PATH = 'imagesA_%i_%i.pth' % (IMG_SIZE, IMG_SIZE)
ATTR_PATH = 'imagesB_%i_%i.pth'


def preprocess_images(img_folder):

    if os.path.isfile(IMG_PATH):
        print("%s exists, nothing to do." % IMG_PATH)
        return

    print("Reading images from {} ...".format(img_folder))
    img_paths = sorted([f for f in os.listdir(img_folder)
                 if os.path.isfile(os.path.join(img_folder, f))
                 & os.path.join(img_folder, f).endswith('.jpg')])
    print("Found {} images".format(len(img_paths)))

    raw_images = []
    for i, img_path in enumerate(img_paths):
        if i % 1000 == 0:
            print(i)
        raw_images.append(mpimg.imread(os.path.join(img_folder,img_path)))

    if len(raw_images) != N_IMAGES:
        raise Exception("Found %i images. Expected %i" % (len(raw_images), N_IMAGES))

    # print("Resizing images ...")
    # all_images = []
    # for i, image in enumerate(raw_images):
    #     if i % 1000 == 0:
    #         print(i)
    #     assert image.shape == (178, 178, 3)
    #     if IMG_SIZE < 178:
    #         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    #     elif IMG_SIZE > 178:
    #         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    #     assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
    #     all_images.append(image)

    data = np.concatenate([img.transpose((2, 0, 1))[None] for img in raw_images], 0)
    data = torch.from_numpy(data)
    assert data.size() == (N_IMAGES, 3, IMG_SIZE, IMG_SIZE)

    print("Saving images to %s ..." % IMG_PATH)
    # torch.save(data[:20000].clone(), 'images_%i_%i_20000.pth' % (IMG_SIZE, IMG_SIZE))
    torch.save(data, IMG_PATH)

    return img_paths


def preprocess_attributes():

    if os.path.isfile(ATTR_PATH):
        print("%s exists, nothing to do." % ATTR_PATH)
        return

    # attr_lines = pd.read_csv('img_attr_merged.csv', encoding='utf-8', sep='\t')
    attr_lines = pd.read_csv('laenge_kleid.csv', encoding='utf-8')
    # attr_lines['muster'] = attr_lines['muster_all-over-muster'] + \
    #                        attr_lines['muster_gebluemt/floral'] + \
    #                        attr_lines['muster_gestreift'] + \
    #                        attr_lines['muster_spitze']
    attr_lines = attr_lines[['img_path', 'laenge']]
    attr_lines['laenge'] = [1 if x >= 4 else 0 for x in attr_lines['laenge']]
    print(attr_lines.head())
    attr_lines = attr_lines.sort_values(['img_path']).set_index('img_path').transpose()
    attributes = dict(zip(attr_lines.index, attr_lines.values.astype(np.bool)))
    print(attributes)

    print("Saving attributes to %s ..." % ATTR_PATH)
    torch.save(attributes, ATTR_PATH)


preprocess_images()
preprocess_attributes()
