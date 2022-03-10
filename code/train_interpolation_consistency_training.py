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
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/ICT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--fold', type=int,
                    default=2, help='cross validation')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
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
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_ratio', type=int, default=5,
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


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    db_train_labeled = BaseDataSets(base_dir=args.root_path, labeled_type="labeled", labeled_ratio=args.labeled_ratio, fold=args.fold, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)]), cross_val=args.cross_val)
    db_train_unlabeled = BaseDataSets(base_dir=args.root_path, labeled_type="unlabeled", labeled_ratio=args.labeled_ratio, fold=args.fold, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)]), cross_val=args.cross_val)

    trainloader_labeled = DataLoader(
        db_train_labeled, batch_size=args.batch_size//2, shuffle=True)
    trainloader_unlabeled = DataLoader(
        db_train_unlabeled, batch_size=args.batch_size//2, shuffle=True)
    logging.info("Labeled slices: {} ".format(len(db_train_labeled)))
    logging.info("Unlabeled slices: {} ".format(len(db_train_unlabeled)))

    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold,
                          split="val", labeled_ratio=args.labeled_ratio, cross_val=args.cross_val)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.DiceLoss(num_classes)

    logging.info("{} iterations per epoch".format(len(trainloader_labeled)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_labeled) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i, data in enumerate(zip(cycle(trainloader_labeled), trainloader_unlabeled)):
            sampled_batch_labeled, sampled_batch_unlabeled = data[0], data[1]

            volume_batch, label_batch = sampled_batch_labeled['image'], sampled_batch_labeled['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = sampled_batch_unlabeled['image'].cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            supervised_loss = 0.5 * \
                (ce_loss(outputs, label_batch[:].long(
                )) + dice_loss(outputs_soft, label_batch[:].unsqueeze(1)))

            if unlabeled_volume_batch.shape[0] != args.batch_size // 2:
                loss = supervised_loss
                consistency_weight = 0.0
                consistency_loss = 0.0
            else:
                # ICT mix factors
                ict_alpha = 0.2
                ict_mix_factors = np.random.beta(
                    ict_alpha, ict_alpha, size=(args.batch_size // 4, 1, 1, 1))
                ict_mix_factors = torch.tensor(
                    ict_mix_factors, dtype=torch.float).cuda()
                unlabeled_volume_batch_0 = unlabeled_volume_batch[0:args.batch_size // 4, ...]
                unlabeled_volume_batch_1 = unlabeled_volume_batch[args.batch_size // 4:, ...]

                # Mix images
                batch_ux_mixed = unlabeled_volume_batch_0 * \
                    (1.0 - ict_mix_factors) + \
                    unlabeled_volume_batch_1 * ict_mix_factors
                # input_volume_batch = torch.cat(
                #     [volume_batch, batch_ux_mixed], dim=0)

                outputs_unlabeled = model(batch_ux_mixed)
                outputs_unlabeled_soft = torch.softmax(
                    outputs_unlabeled, dim=1)

                with torch.no_grad():
                    ema_output_ux0 = torch.softmax(
                        ema_model(unlabeled_volume_batch_0), dim=1)
                    ema_output_ux1 = torch.softmax(
                        ema_model(unlabeled_volume_batch_1), dim=1)
                    batch_pred_mixed = ema_output_ux0 * \
                        (1.0 - ict_mix_factors) + \
                        ema_output_ux1 * ict_mix_factors

                consistency_weight = get_current_consistency_weight(
                    iter_num // (args.max_iterations/args.consistency_rampup))
                consistency_loss = torch.mean(
                    (outputs_unlabeled_soft - batch_pred_mixed) ** 2)
                loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', supervised_loss, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f' %
                (iter_num, loss.item(), supervised_loss.item()))

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            
        save_latest = os.path.join(
            snapshot_path, '{}_latest_model.pth'.format(args.model))
        torch.save(model.state_dict(), save_latest)

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
