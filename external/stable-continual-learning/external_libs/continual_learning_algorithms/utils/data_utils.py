# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Define utility functions for manipulating datasets
"""
import os
import numpy as np
import sys
from copy import deepcopy

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tarfile
import zipfile
import random
import cv2
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IMG_MEAN = np.array((103.94,116.78,123.68), dtype=np.float32)
############################################################
### Data augmentation utils ################################
############################################################
def image_scaling(images):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.
    Args:
        images: Training images to scale.
    """
    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(images)[1]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(images)[2]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    images = tf.image.resize_images(images, new_shape)
    result = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
    return result


def random_crop_and_pad_image(images, crop_h, crop_w):
    """
    Randomly crop and pads the input images.
    Args:
      images: Training i mages to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
    """
    image_shape = tf.shape(images)
    image_pad = tf.image.pad_to_bounding_box(images, 0, 0, tf.maximum(crop_h, image_shape[1]), tf.maximum(crop_w, image_shape[2]))
    img_crop = tf.map_fn(lambda img: tf.random_crop(img, [crop_h,crop_w,3]), image_pad)
    return img_crop

def random_horizontal_flip(x):
    """
    Randomly flip a batch of images horizontally
    Args:
    x               Tensor of shape B x H x W x C
    Returns:
    random_flipped  Randomly flipped tensor of shape B x H x W x C
    """
    # Define random horizontal flip
    flips = [(slice(None, None, None), slice(None, None, random.choice([-1, None])), slice(None, None, None))
            for _ in xrange(x.shape[0])]
    random_flipped = np.array([img[flip] for img, flip in zip(x, flips)])
    return random_flipped

############################################################
### CIFAR download utils ###################################
############################################################
CIFAR_10_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_100_URL = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR_10_DIR = "/cifar_10"
CIFAR_100_DIR = "/cifar_100"

def construct_split_cifar(task_labels, is_cifar_100=True):
    """
    Construct Split CIFAR-10 and CIFAR-100 datasets

    Args:
        task_labels     Labels of different tasks
        data_dir        Data directory where the CIFAR data will be saved
    """

    data_dir = 'CIFAR_data'

    # Get the cifar dataset
    cifar_data = _get_cifar(data_dir, is_cifar_100)

    # Define a list for storing the data for different tasks
    datasets = []

    # Data splits
    sets = ["train", "validation", "test"]

    for task in task_labels:

        for set_name in sets:
            this_set = cifar_data[set_name]

            global_class_indices = np.column_stack(np.nonzero(this_set[1]))
            count = 0

            for cls in task:
                if count == 0:
                    class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] ==
                                                                    cls][:,np.array([True, False])])
                else:
                    class_indices = np.append(class_indices, np.squeeze(global_class_indices[global_class_indices[:,1] ==\
                                                                                 cls][:,np.array([True, False])]))

                count += 1

            class_indices = np.sort(class_indices, axis=None)

            if set_name == "train":
                train = {
                    'images':deepcopy(this_set[0][class_indices, :]),
                    'labels':deepcopy(this_set[1][class_indices, :]),
                }
            elif set_name == "validation":
                validation = {
                    'images':deepcopy(this_set[0][class_indices, :]),
                    'labels':deepcopy(this_set[1][class_indices, :]),
                }
            elif set_name == "test":
                test = {
                    'images':deepcopy(this_set[0][class_indices, :]),
                    'labels':deepcopy(this_set[1][class_indices, :]),
                }

        cifar = {
            'train': train,
            'validation': validation, 
            'test': test,
        }

        datasets.append(cifar)

    return datasets


def _get_cifar(data_dir, is_cifar_100):
    """
    Get the CIFAR-10 and CIFAR-100 datasets

    Args:
        data_dir        Directory where the downloaded data will be stored
    """
    x_train = None
    y_train = None
    x_validation = None
    y_validation = None
    x_test = None
    y_test = None
    l = None

    # Download the dataset if needed
    _cifar_maybe_download_and_extract(data_dir)

    # Dictionary to store the dataset
    dataset = dict()
    dataset['train'] = []
    dataset['validation'] = []
    dataset['test'] = []

    def dense_to_one_hot(labels_dense, num_classes=100):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    if is_cifar_100:
        # Load the training data of CIFAR-100
        f = open(data_dir + CIFAR_100_DIR + '/train', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()
    
        _X = datadict['data']
        _Y = np.array(datadict['fine_labels'])
        _Y = dense_to_one_hot(_Y, num_classes=100)

        _X = np.array(_X, dtype=float) / 255.0
        _X = _X.reshape([-1, 3, 32, 32])
        _X = _X.transpose([0, 2, 3, 1])
    
        # Compute the data mean for normalization
        x_train_mean = np.mean(_X, axis=0)

        x_train = _X[:40000]
        y_train = _Y[:40000]

        x_validation = _X[40000:]
        y_validation = _Y[40000:]
    else:
        # Load all the training batches of the CIFAR-10
        for i in range(5):
            f = open(data_dir + CIFAR_10_DIR + '/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()
            
            _X = datadict['data']
            _Y = np.array(datadict['labels'])
            _Y = dense_to_one_hot(_Y, num_classes=10)
            
            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            
            if x_train is None:
                x_train = _X
                y_train = _Y
            else:
                x_train = np.concatenate((x_train, _X), axis=0)
                y_train = np.concatenate((y_train, _Y), axis=0)
    
        # Compute the data mean for normalization
        x_train_mean = np.mean(x_train, axis=0)
        x_validation = x_train[:40000] # We don't use validation set with CIFAR-10
        y_validation = y_train[40000:]

    # Normalize the train and validation sets
    x_train -= x_train_mean
    x_validation -= x_train_mean

    dataset['train'].append(x_train)
    dataset['train'].append(y_train)
    dataset['train'].append(l)

    dataset['validation'].append(x_validation)
    dataset['validation'].append(y_validation)
    dataset['validation'].append(l)

    if is_cifar_100:
        # Load the test batch of CIFAR-100
        f = open(data_dir + CIFAR_100_DIR + '/test', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()
    
        _X = datadict['data']
        _Y = np.array(datadict['fine_labels'])
        _Y = dense_to_one_hot(_Y, num_classes=100)
    else:
        # Load the test batch of CIFAR-10
        f = open(data_dir + CIFAR_10_DIR + '/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        _X = datadict["data"]
        _Y = np.array(datadict['labels'])
        _Y = dense_to_one_hot(_Y, num_classes=10)

    _X = np.array(_X, dtype=float) / 255.0
    _X = _X.reshape([-1, 3, 32, 32])
    _X = _X.transpose([0, 2, 3, 1])

    x_test = _X
    y_test = _Y

    # Normalize the test set
    x_test -= x_train_mean

    dataset['test'].append(x_test)
    dataset['test'].append(y_test)
    dataset['test'].append(l)

    return dataset


def _print_download_progress(count, block_size, total_size):
    """
    Show the download progress of the cifar data
    """
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def _cifar_maybe_download_and_extract(data_dir):
    """
    Routine to download and extract the cifar dataset

    Args:
        data_dir      Directory where the downloaded data will be stored
    """
    cifar_10_directory = data_dir + CIFAR_10_DIR
    cifar_100_directory = data_dir + CIFAR_100_DIR

    # If the data_dir does not exist, create the directory and download
    # the data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

        url = CIFAR_10_URL
        filename = url.split('/')[-1]
        file_path = os.path.join(data_dir, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(data_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)
        print("Done.")

        url = CIFAR_100_URL
        filename = url.split('/')[-1]
        file_path = os.path.join(data_dir, filename)
        zip_cifar_100 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(data_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)
        print("Done.")

        os.rename(data_dir + "/cifar-10-batches-py", cifar_10_directory)
        os.rename(data_dir + "/cifar-100-python", cifar_100_directory)
        os.remove(zip_cifar_10)
        os.remove(zip_cifar_100)


#########################################
## MNIST Utils ##########################
#########################################
def reformat_mnist(datasets):
    """
    Routine to Reformat the mnist dataset into a 3d tensor
    """
    image_size = 28 # Height of MNIST dataset
    num_channels = 1 # Gray scale
    for i in range(len(datasets)):
        sets = ["train", "validation", "test"]
        for set_name in sets:
            datasets[i]['%s'%set_name]['images'] = datasets[i]['%s'%set_name]['images'].reshape\
            ((-1, image_size, image_size, num_channels)).astype(np.float32)

    return datasets


def rotate_image_by_angle(img, angle=45):
    WIDTH, HEIGHT = 28 , 28
    img = img.reshape((WIDTH, HEIGHT))
    img = ndimage.rotate(img, angle, reshape=False, order=0)
    out = np.array(img).flatten()
    return out

def construct_rotate_mnist(num_tasks):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    datasets = []

    for i in range(num_tasks):
        per_task_rotation = 180.0 / num_tasks
        rotation_degree = (i - 1)*per_task_rotation
        rotation_degree -= (np.random.random()*per_task_rotation)
        copied_mnist = deepcopy(mnist)
        sets = ["train", "validation", "test"]
        for set_name in sets:
            this_set = getattr(copied_mnist, set_name) # shallow copy

            rotate_image_by_angle(this_set._images[0])
            this_set._images = np.array([rotate_image_by_angle(img, rotation_degree) for img in this_set._images])
            if set_name == "train":
                train = { 
                    'images':this_set._images,
                    'labels':this_set.labels,
                }
            elif set_name == "validation":
                validation = {
                    'images':this_set._images,
                    'labels':this_set.labels,
                }
            elif set_name == "test":
                test = {
                    'images':this_set._images,
                    'labels':this_set.labels,
                }
        dataset = {
            'train': train,
            'validation': validation,
            'test': test,
        }

        datasets.append(dataset)

    return datasets
def construct_permute_mnist(num_tasks):
    """
    Construct a dataset of permutted mnist images

    Args:
        num_tasks   Number of tasks
    Returns
        dataset     A permutted mnist dataset
    """
    # Download and store mnist dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    datasets = []

    for i in range(num_tasks):
        perm_inds = list(range(mnist.train.images.shape[1]))
        np.random.shuffle(perm_inds)
        copied_mnist = deepcopy(mnist)
        sets = ["train", "validation", "test"]
        for set_name in sets:
            this_set = getattr(copied_mnist, set_name) # shallow copy
            this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
            # print(this_set._images.shape)
            if set_name == "train":
                train = { 
                    'images':this_set._images,
                    'labels':this_set.labels,
                }
            elif set_name == "validation":
                validation = {
                    'images':this_set._images,
                    'labels':this_set.labels,
                }
            elif set_name == "test":
                test = {
                    'images':this_set._images,
                    'labels':this_set.labels,
                }
        dataset = {
            'train': train,
            'validation': validation,
            'test': test,
        }

        datasets.append(dataset)

    return datasets

def construct_split_mnist(task_labels):
    """
    Construct a split mnist dataset

    Args:
        task_labels     List of split labels

    Returns:
        dataset         A list of split datasets

    """
    # Download and store mnist dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    datasets = []

    sets = ["train", "validation", "test"]

    for task in task_labels:

        for set_name in sets:
            this_set = getattr(mnist, set_name)

            global_class_indices = np.column_stack(np.nonzero(this_set.labels))
            count = 0

            for cls in task:
                if count == 0:
                    class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] ==\
                                                                    cls][:,np.array([True, False])])
                else:
                    class_indices = np.append(class_indices, np.squeeze(global_class_indices[global_class_indices[:,1] ==\
                                                                                             cls][:,np.array([True, False])]))
                count += 1

            class_indices = np.sort(class_indices, axis=None)

            if set_name == "train":
                train = {
                    'images':deepcopy(mnist.train.images[class_indices, :]),
                    'labels':deepcopy(mnist.train.labels[class_indices, :]),
                }
            elif set_name == "validation":
                validation = {
                    'images':deepcopy(mnist.validation.images[class_indices, :]),
                    'labels':deepcopy(mnist.validation.labels[class_indices, :]),
                }
            elif set_name == "test":
                test = {
                    'images':deepcopy(mnist.test.images[class_indices, :]),
                    'labels':deepcopy(mnist.test.labels[class_indices, :]),
                }

        mnist2 = {
            'train': train,
            'validation': validation,
            'test': test,
        }

        datasets.append(mnist2)

    return datasets

###################################################
###### ImageNet Utils #############################
###################################################
def construct_split_imagenet(task_labels, data_dir):
    """
    Construct Split ImageNet dataset

    Args:
        task_labels     Labels of different tasks
        data_dir        Data directory from where to load the imagenet data
    """

    # Load the imagenet dataset
    imagenet_data = _load_imagenet(data_dir)

    # Define a list for storing the data for different tasks
    datasets = []

    # Data splits
    sets = ["train", "test"]

    for task in task_labels:

        for set_name in sets:
            this_set = imagenet_data[set_name]

            global_class_indices = np.column_stack(np.nonzero(this_set[1]))
            count = 0

            for cls in task:
                if count == 0:
                    class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] ==
                                                                    cls][:,np.array([True, False])])
                else:
                    class_indices = np.append(class_indices, np.squeeze(global_class_indices[global_class_indices[:,1] ==\
                                                                                 cls][:,np.array([True, False])]))

                count += 1

            class_indices = np.sort(class_indices, axis=None)

            if set_name == "train":
                train = {
                    'images':deepcopy(this_set[0][class_indices, :]),
                    'labels':deepcopy(this_set[1][class_indices, :]),
                }
            elif set_name == "test":
                test = {
                    'images':deepcopy(this_set[0][class_indices, :]),
                    'labels':deepcopy(this_set[1][class_indices, :]),
                }

        imagenet = {
            'train': train,
            'test': test,
        }

        datasets.append(imagenet)

    return datasets

def _load_imagenet(data_dir):
    """
    Load the ImageNet data

    Args:
        data_dir    Directory where the pickle files have been dumped
    """
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    # Dictionary to store the dataset
    dataset = dict()
    dataset['train'] = []
    dataset['test'] = []

    def dense_to_one_hot(labels_dense, num_classes=100):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    # Load the training batches
    for i in range(4):
        f = open(data_dir + '/train_batch_' + str(i), 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        _X = datadict['data']
        _Y = np.array(datadict['labels'])

        # Convert the lables to one-hot
        _Y = dense_to_one_hot(_Y)

        # Normalize the images
        _X = np.array(_X, dtype=float)/ 255.0
        _X = _X.reshape([-1, 224, 224, 3])

        if x_train is None:
            x_train = _X
            y_train = _Y
        else:
            x_train = np.concatenate((x_train, _X), axis=0)
            y_train = np.concatenate((y_train, _Y), axis=0)

    dataset['train'].append(x_train)
    dataset['train'].append(y_train)

    # Load test batches
    for i in range(4):
        f = open(data_dir + '/test_batch_' + str(i), 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        _X = datadict['data']
        _Y = np.array(datadict['labels'])

        # Convert the lables to one-hot
        _Y = dense_to_one_hot(_Y)

        # Normalize the images
        _X = np.array(_X, dtype=float)/ 255.0
        _X = _X.reshape([-1, 224, 224, 3])

        if x_test is None:
            x_test = _X
            y_test = _Y
        else:
            x_test = np.concatenate((x_test, _X), axis=0)
            y_test = np.concatenate((y_test, _Y), axis=0)

    dataset['test'].append(x_test)
    dataset['test'].append(y_test)


    return dataset

if __name__ == "__main__":
    construct_rotate_mnist(20)
    # rotate_image_by_angle(np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
