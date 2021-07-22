import argparse
import os
import shutil
from glob import glob
import numpy

import torch

from networks.unet_3D_dv_semi import unet_3D_dv_semi
from networks.unet_3D import unet_3D
from test_urpc_util import test_all_case


def net_factory(net_type="unet_3D", num_classes=3, in_channels=1):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=num_classes, in_channels=in_channels).cuda()
    elif net_type == "unet_3D_dv_semi":
        net = unet_3D_dv_semi(n_classes=num_classes,
                              in_channels=in_channels).cuda()
    else:
        net = None
    return net


def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "../model/{}/Prediction".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(FLAGS.model, num_classes, in_channels=1)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/BraTS2019', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default="BraTS2019/Uncertainty_Rectified_Pyramid_Consistency_25_labeled", help='experiment_name')
    parser.add_argument('--model', type=str,
                        default="unet_3D_dv_semi", help='model_name')
    FLAGS = parser.parse_args()

    metric = Inference(FLAGS)
    print(metric)
