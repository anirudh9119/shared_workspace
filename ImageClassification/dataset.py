from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse
import multiprocessing

import numpy as np
#import tensorflow as tf

from functools import partial
from skimage.filters import gaussian
import torch
from tqdm import tqdm

def get_distance(x, y, px, py):
    "Simple distance formula."
    return np.sqrt((px - x)**2 + (py - y)**2)

def get_points(num_points, xmin, ymin, xmax, ymax, min_distance):
    "Sample points from a 2D range with a minimum distance using rejection sampling."
    points = []

    for _ in range(num_points):
        while True:
            x = np.random.randint(xmin, xmax)
            y = np.random.randint(ymin, ymax)

            rejected = False
            for px, py in points:
                d = get_distance(x, y, px, py)
                if d < min_distance:
                    rejected = True
                    break

            if rejected:
                continue

            points.append((x, y))
            break

    return points

def generate_sample(dataset_images, dataset_labels, image_size, digit_size, max_digits, sigma):
    "Construct a single counting MNIST sample."
    digit_size_half = digit_size // 2
    num_digits = max_digits

    image = np.zeros(
        shape=(image_size, image_size, 1),
        dtype=np.float64)
    density = np.zeros(
        shape=(image_size, image_size, 1),
        dtype=np.float64)


    ids = np.random.choice(np.arange(len(dataset_images)), size=num_digits)
    digits = dataset_images[ids]
    labels = dataset_labels[ids]

    points = get_points(
        num_points=num_digits,
        xmin=digit_size_half,
        ymin=digit_size_half,
        xmax=image_size - digit_size_half,
        ymax=image_size - digit_size_half,
        min_distance=digit_size_half)

    count = np.zeros(10)
    for i, (digit, label, (x, y)) in enumerate(zip(digits, labels, points)):
        xmin = x - digit_size_half
        xmax = x + digit_size_half
        ymin = y - digit_size_half
        ymax = y + digit_size_half

        image[ymin:ymax, xmin:xmax] += digit

        #if label % 2 == 0:
        #    density[y, x] = 1
        #    count += 1
        count[label] += 1.
    image = np.clip(image, 0.0, 1.0)
    #image = image

    density = gaussian(density, sigma=sigma, mode='constant')
    density = density.astype(np.float32)

    return image, density, count


def extract_mnist(data_dir):

    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1))

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_label = np.asarray(loaded[8:].reshape((num_mnist_train)))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1))

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_label = np.asarray(loaded[8:].reshape((num_mnist_test)))

    return np.concatenate((train_image, test_image)), \
        np.concatenate((train_label, test_label))



class CountingMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, split = 'train', path = 'MNIST', max_digits = 3, num_examples = 100000):
        data, labels = extract_mnist(path)
        if split == 'train':
            images = data[:60000]
            labels = labels[:60000]
        else:
            images = data[60000:]
            labels = labels[60000:]

        self.images = []
        self.count = []
        print('preparing dataset')
        for k in tqdm(range(num_examples)):
            image,_, count = generate_sample(images, labels, 100, 28, max_digits, 5)
            self.images.append(image)
            self.count.append(count)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]).float().permute(2, 0, 1), torch.from_numpy(self.count[idx]).float()


if __name__ == '__main__':
    data, labels = extract_mnist('MNIST')
    train_images = data[:60000]
    train_labels = labels[:60000]

    image, density, count = generate_sample(train_images, train_labels, 100, 28, 5, 5)

    import cv2
    print(count)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()