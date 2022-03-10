import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage import exposure
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset
import copy

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, labeled_type="labeled", labeled_ratio=10, split='train', transform=None, fold=1, cross_val=True):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.labeled_type = labeled_type
        self.all_volumes = sorted(os.listdir(self._base_dir + "/all_volumes"))
        if cross_val:  # 5-fold  cross validation
            train_ids, val_ids = self._get_fold_ids(fold)
        else:
            train_ids, val_ids, _ = self._get_split_ids()
        train_ids = sorted(train_ids)
        all_labeled_ids = train_ids[::labeled_ratio]
        if self.split == 'train':
            self.all_slices = os.listdir(self._base_dir + "/all_slices")
            self.sample_list = []
            labeled_ids = [i for i in all_labeled_ids if i in train_ids]
            unlabeled_ids = [i for i in train_ids if i not in labeled_ids]
            if self.labeled_type == "labeled":
                print("Labeled patients IDs", labeled_ids)
                for ids in labeled_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids.replace(".h5", "")), x) != None, self.all_slices))
                    self.sample_list.extend(new_data_list)
                print("total labeled {} samples".format(len(self.sample_list)))
            else:
                print("Unlabeled patients IDs", unlabeled_ids)
                for ids in unlabeled_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids.replace(".h5", "")), x) != None, self.all_slices))
                    self.sample_list.extend(new_data_list)
                print("total unlabeled {} samples".format(len(self.sample_list)))

        elif self.split == 'val':
            print("val_ids", val_ids)
            self.sample_list = []
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids.replace(".h5", "")), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

    def _get_fold_ids(self, fold):
        folds = KFold(n_splits=5, shuffle=False)
        all_cases = np.array(self.all_volumes)
        k_fold_data = []
        for trn_idx, val_idx in folds.split(all_cases):
            k_fold_data.append([all_cases[trn_idx], all_cases[val_idx]])
        train_set = k_fold_data[fold][0]
        test_set = k_fold_data[fold][1]
        return train_set, test_set

    def _get_split_ids(self):
        all_cases = np.array(self.all_volumes)
        rest_set, test_set = train_test_split(all_cases, test_size=int(
            len(self.all_volumes)*0.2), shuffle=True, random_state=1234)
        train_set, val_set = train_test_split(rest_set, test_size=int(
            len(self.all_volumes)*0.1), shuffle=True, random_state=1234)
        print("test_set", sorted(test_set))
        return train_set, val_set, test_set

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/all_slices/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/all_volumes/{}".format(case), 'r')
        if self.split == "train":
            image = h5f['image'][:]
            label = h5f["label"][:]
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:].astype(np.int16)
            sample = {'image': image, 'label': label}
        sample["idx"] = case.split("_")[0]
        return sample


def random_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label


def random_noise(image, label, mu=0, sigma=0.1):
    noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]),
                    -2 * sigma, 2 * sigma)
    noise = noise + mu
    image = image + noise
    return image, label


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t**(n-i)) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, label, prob=0.5):
    if random.random() >= prob:
        return x, label
    points = [[0, 0], [random.random(), random.random()], [
        random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label


def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label


class RandomGenerator_Strong_Weak(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label, cval=0)
        if random.random() > 0.5:
            image, label = random_noise(image, label)

        x, y = image.shape
        image_w = copy.deepcopy(image)
        image_w = zoom(
            image_w, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        if random.random() > 0.33:
            image, label = nonlinear_transformation(image, label)
        elif random.random() < 0.66 and random.random() > 0.33:
            image, label = random_rescale_intensity(image, label)
        else:
            image, label = random_equalize_hist(image, label)
        image_s = image
        image_s = zoom(
            image_s, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image_w = torch.from_numpy(
            image_w.astype(np.float32)).unsqueeze(0)
        image_s = torch.from_numpy(
            image_s.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int16))
        sample = {'image_w': image_w, 'image_s': image_s, 'label': label}
        return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label, cval=0)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        if random.random() > 0.33:
            image, label = nonlinear_transformation(image, label)
        elif random.random() < 0.66 and random.random() > 0.33:
            image, label = random_rescale_intensity(image, label)
        elif random.random() > 0.66:
            image, label = random_equalize_hist(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int16))
        sample = {'image': image, 'label': label}
        return sample
