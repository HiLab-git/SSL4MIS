import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import glob
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')


# For Transformer
parser.add_argument('--patch_size', type=list,  default=[224, 224], help='patch size of network input')
parser.add_argument('--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+',  )
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

##

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS, patch_size=[256, 256]):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds" or FLAGS.model == "unet_urpc" or FLAGS.model == "unet_cct":
                out_main, _, _, _ = net(input)
            elif FLAGS.model == "unet_seg" or FLAGS.model == "unet_dtc" or FLAGS.model == "unet_sdf":
                _ , out_main = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    
    
    model_paths = glob.glob(os.path.join(snapshot_path, "*best*"))
    if len(model_paths)>0:
        os.makedirs(test_save_path)
    elif len(model_paths)==0:
        print("no models in {}", snapshot_path)
    avg_metrics = []
    for save_mode_path in model_paths:
        #save_mode_path = os.path.join(snapshot_path, model_path)
        try:
            net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
            net.load_state_dict(torch.load(save_mode_path))
        except:
            config = get_config(FLAGS)
            net = ViT_seg(config, img_size=FLAGS.patch_size, num_classes=FLAGS.num_classes).cuda()
            net.load_from(config)
            net.load_state_dict(torch.load(save_mode_path))

        print("init weight from {}".format(save_mode_path))
        net.eval()

        first_total = 0.0
        second_total = 0.0
        third_total = 0.0
        for case in tqdm(image_list):
            if "ViT" in FLAGS.model:
                first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS, patch_size=[224, 224])
            else:
                first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
            first_total += np.asarray(first_metric)
            second_total += np.asarray(second_metric)
            third_total += np.asarray(third_metric)
        avg_metric = [first_total / len(image_list), second_total /
                      len(image_list), third_total / len(image_list)]

        print(avg_metric)
        print((avg_metric[0]+avg_metric[1]+avg_metric[2])/3)
        avg_metrics.append(avg_metric)
        del net
        
    return avg_metrics


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metrics = Inference(FLAGS)
    for metric in metrics:
        print(metric)
        print((metric[0]+metric[1]+metric[2])/3)
