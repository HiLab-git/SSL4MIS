import argparse
import logging
import os
import random
import shutil
import sys
import time
from itertools import cycle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.discriminator import FCDiscriminator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ProstateX', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ProstateX/DCT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--fold', type=int,
                    default=1, help='cross validation')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--cross_val', type=bool,
                    default=True, help='5-fold cross validation or random split 7/1/2 for training/validation/testing')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.03,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_ratio', type=int, default=8,
                    help='1/labeled_ratio data is provided mask')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    model1 = net_factory(net_type=args.model, in_chns=1,
                         class_num=num_classes)
    model2 = net_factory(net_type=args.model, in_chns=1,
                         class_num=num_classes)

    db_train_labeled = BaseDataSets(base_dir=args.root_path, labeled_type="labeled", labeled_ratio=args.labeled_ratio, fold=args.fold, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)]), cross_val=args.cross_val)
    db_train_unlabeled = BaseDataSets(base_dir=args.root_path, labeled_type="unlabeled", labeled_ratio=args.labeled_ratio, fold=args.fold, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)]), cross_val=args.cross_val)
    logging.info("Labeled slices: {} ".format(len(db_train_labeled)))
    logging.info("Unlabeled slices: {} ".format(len(db_train_unlabeled)))

    trainloader_labeled = DataLoader(
        db_train_labeled, batch_size=args.batch_size//2, shuffle=True)
    trainloader_unlabeled = DataLoader(
        db_train_unlabeled, batch_size=args.batch_size//2, shuffle=True)

    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold,
                          split="val", labeled_ratio=args.labeled_ratio, cross_val=args.cross_val)
    valloader = DataLoader(db_val, batch_size=1)

    model1.train()
    model2.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    logging.info("{} iterations per epoch".format(len(trainloader_unlabeled)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_unlabeled) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i, (sampled_batch_labeled, sampled_batch_unlabeled) in enumerate(zip(cycle(trainloader_labeled), trainloader_unlabeled)):
            volume_batch, label_batch = sampled_batch_labeled['image'], sampled_batch_labeled['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = sampled_batch_unlabeled['image'].cuda()

            outputs1 = model1(volume_batch)
            outputs1_soft = torch.softmax(outputs1, dim=1)

            outputs1_unlabeled = model1(unlabeled_volume_batch)
            outputs1_unlabeled_soft = torch.softmax(outputs1_unlabeled, dim=1)

            outputs2 = model2(volume_batch)
            outputs2_soft = torch.softmax(outputs2, dim=1)

            outputs2_unlabeled = model2(unlabeled_volume_batch)
            outputs2_unlabeled_soft = torch.softmax(outputs2_unlabeled, dim=1)

            supervised_loss1 = 0.5 * \
                (ce_loss(outputs1, label_batch[:].long(
                )) + dice_loss(outputs1_soft, label_batch[:].unsqueeze(1)))
            supervised_loss2 = 0.5 * \
                (ce_loss(outputs2, label_batch[:].long(
                )) + dice_loss(outputs2_soft, label_batch[:].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(
                outputs1_unlabeled_soft.detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(
                outputs2_unlabeled_soft.detach(), dim=1, keepdim=False)

            pseudo_supervision1 = ce_loss(outputs1_unlabeled, pseudo_outputs2)
            pseudo_supervision2 = ce_loss(outputs2_unlabeled, pseudo_outputs1)

            consistency_weight = get_current_consistency_weight(
                iter_num // (args.max_iterations/args.consistency_rampup))

            model1_loss = supervised_loss1 + consistency_weight * pseudo_supervision1
            model2_loss = supervised_loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[0, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice',
                                  performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95',
                                  mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice',
                                  performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95',
                                  mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        save_latest = os.path.join(
            snapshot_path, '{}_latest_model.pth'.format(args.model))
        torch.save(model1.state_dict(), save_latest)

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.cross_val:
        snapshot_path = "../model/{}/1_of_{}_labeled/fold{}/{}".format(
            args.exp, args.labeled_ratio, args.fold, args.model)
    else:
        snapshot_path = "../model/{}/1_of_{}_labeled/{}".format(
            args.exp, args.labeled_ratio, args.model)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
