import argparse
import os
import re
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from matplotlib.pyplot import axis
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/FullSup', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--best_model', type=int,
                    default=1, help='the best or latest checkpoints')
parser.add_argument('--labeled_ratio', type=int, default=10,
                    help='1/labeled_ratio data is provided mask')
parser.add_argument('--fold', type=int,
                    default=3, help='fold')
parser.add_argument('--cross_val', type=int,
                    default=0, help='5-fold cross validation or random split 7/1/2 for training/validation/testing')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="label",
                    help='label')


def get_fold_ids(FLAGS):
    all_volumes = os.listdir(FLAGS.root_path + "/all_volumes")
    folds = KFold(n_splits=5, shuffle=False)
    all_cases = np.array(all_volumes)
    k_fold_data = []
    for trn_idx, val_idx in folds.split(all_cases):
        k_fold_data.append([all_cases[trn_idx], all_cases[val_idx]])
    return sorted(k_fold_data[FLAGS.fold][0]), sorted(k_fold_data[FLAGS.fold][1])


def get_split_ids(FLAGS):
    all_cases = sorted(np.array(os.listdir(FLAGS.root_path + "/all_volumes")))
    rest_set, test_set = train_test_split(all_cases, test_size=int(
        len(all_cases)*0.2), shuffle=True, random_state=1234)
    train_set, val_set = train_test_split(rest_set, test_size=int(
        len(all_cases)*0.1), shuffle=True, random_state=1234)
    return sorted(train_set), sorted(val_set), sorted(test_set)

def calculate_metric_percase(pred, gt, spacing):
    if pred.sum() > 0 and gt.sum() > 0:
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    else:
        dice = 0.0
        hd95 = 100.0
        asd = 20.0
    return dice, hd95, asd


# def test_single_volume(case, net, test_save_path, FLAGS):
#     h5f = h5py.File(FLAGS.root_path +
#                     "/all_volumes/{}".format(case), 'r')
#     image = h5f['image'][:]
#     label = h5f['label'][:]
#     spacing = h5f['spacing'][:]
#     prediction = np.zeros_like(label)
#     for ind in range(image.shape[0]):
#         slice = image[ind, :, :]
#         x, y = slice.shape[0], slice.shape[1]
#         slice = zoom(slice, (FLAGS.patch_size / x, FLAGS.patch_size / y), order=0)
#         input = torch.from_numpy(slice).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out_main = net(input)
#             out = torch.argmax(torch.softmax(
#                 out_main, dim=1), dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / FLAGS.patch_size, y / FLAGS.patch_size), order=0)
#             prediction[ind] = pred
#     case = case.replace(".h5", "")

#     metric_list = []
#     for i in range(1, FLAGS.num_classes):
#         metric_list.append(calculate_metric_percase(
#             prediction == i, label == i, spacing=(spacing[2], spacing[0], spacing[1])))
#     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     img_itk.SetSpacing(spacing)
#     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     prd_itk.SetSpacing(spacing)
#     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     lab_itk.SetSpacing(spacing)
#     sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
#     sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
#     sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
#     return np.array(metric_list)


def test_single_volume(case, net, test_save_path, FLAGS, batch_size=12):
    h5f = h5py.File(FLAGS.root_path +
                    "/all_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    spacing = h5f['spacing'][:]
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        ind_x = np.array([i for i in range(image.shape[0])])
        for ind in ind_x[::batch_size]:
            if ind + batch_size < image.shape[0]:
                slice = image[ind:ind + batch_size, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        net(input), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:ind + batch_size, ...] = pred
            else:
                slice = image[ind:, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(slice, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        net(input), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (1, x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
                    prediction[ind:, ...] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    case = case.replace(".h5", "")

    metric_list = []
    for i in range(1, FLAGS.num_classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i, spacing=(spacing[2], spacing[0], spacing[1])))
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing(spacing)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing(spacing)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing(spacing)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return np.array(metric_list)


def Inference(FLAGS):
    if FLAGS.cross_val:
        _, test_ids = get_fold_ids(FLAGS)
    else:
        _, _, test_ids = get_split_ids(FLAGS)
    print(test_ids)
    all_volumes = sorted(os.listdir(FLAGS.root_path + "/all_volumes"))
    image_list = []
    for ids in test_ids:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, all_volumes))
        image_list.extend(new_data_list)

    if FLAGS.cross_val:
        snapshot_path = "../model/{}/1_of_{}_labeled/fold{}/{}".format(
            FLAGS.exp, FLAGS.labeled_ratio, FLAGS.fold, FLAGS.model)
        test_save_path = "../model/{}/1_of_{}_labeled/fold{}/{}/prediction/".format(
            FLAGS.exp, FLAGS.labeled_ratio, FLAGS.fold, FLAGS.model)

    else:
        snapshot_path = "../model/{}/1_of_{}_labeled/{}".format(
            FLAGS.exp, FLAGS.labeled_ratio, FLAGS.model)
        test_save_path = "../model/{}/1_of_{}_labeled/{}/prediction/".format(
            FLAGS.exp, FLAGS.labeled_ratio, FLAGS.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    if FLAGS.best_model:
        save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    else:
        save_mode_path = os.path.join(snapshot_path, '{}_latest_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    metric_array = np.zeros((len(image_list), FLAGS.num_classes-1, 3))
    for ind, case in enumerate(tqdm(image_list)):
        print(case)
        cases_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        print(cases_metric)
        metric_array[ind, ...] = cases_metric
    np.save(test_save_path+"/Results.npy", metric_array)
    return metric_array


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    print("Inference fold{}".format(FLAGS.fold))
    metric_array = Inference(FLAGS)
    print("mean class results:", np.mean(metric_array, axis=0))
    print("mean case results:", np.mean(metric_array, axis=0).mean(axis=0))

 