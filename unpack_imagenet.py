#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Converts pickled imagenet_32x32 files to .npy files.
The default imagenet_32x32 data files are stored in Python 3
pickly encoding.
The rest of our code is in Python 2, so we have a separate script that
just deals with this issue separately.
You can execute it as:
python3 convert_imagenet.py
after which you should no longer have to manually deal with imagenet.
"""
import os
import numpy as np
import pickle

from PIL import Image
import matplotlib.pyplot as plt

_DATA_DIR = "data/imagenet32/train_32x32/"


def unpickle(filename):
    print(filename)
    with open(filename, "rb") as fo:
        dict = pickle.load(fo)
    return dict


train_file_names = ["train_data_batch_" + str(idx) for idx in range(1, 11)]
val_file_names = ["val_data"]
img_size = 32
for file_name in train_file_names + val_file_names:
    data = unpickle(os.path.join(_DATA_DIR, file_name))

    image_file_name = file_name + "_image"
    label_file_name = file_name + "_label"

    x = data["data"]
    y = data["labels"]
    x = x/np.float32(255)

    # mean_image = data['mean']
    # mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    # center images to 0 mean #TODO move to pre-processing
    # x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2 * img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    # plt.imshow(x[22, :, :, :].transpose(2, 1, 0))
    # plt.show()

    np.save(os.path.join(_DATA_DIR, image_file_name), data["data"])
    np.save(os.path.join(_DATA_DIR, label_file_name), np.array(data["labels"]))
