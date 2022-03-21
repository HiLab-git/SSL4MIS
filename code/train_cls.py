import argparse
import logging
import os
import re
import random
import shutil
import sys
import time
from xml.etree.ElementInclude import default_loader

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributions import Categorical
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (
    BaseDataSets,
    RandomGenerator,
    TwoStreamBatchSampler,
    WeakStrongAugment,
)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path", type=str, default="../data/ACDC", help="Name of Experiment"
)
parser.add_argument("--exp", type=str, default="ACDC/CLS_1", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument(
    "--max_iterations", type=int, default=30000, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
)
parser.add_argument(
    "--patch_size", type=list, default=[256, 256], help="patch size of network input"
)
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument(
    "--num_classes", type=int, default=4, help="output channel of network"
)
parser.add_argument(
    "--load", default=False, action="store_true", help="restore previous checkpoint"
)
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.5,
    help="confidence threshold for using pseudo-labels",
)
parser.add_argument(
    "--self_coeff",
    type=float,
    default=2,
    help="tradeoff coefficient for self-labelling loss",
)
parser.add_argument(
    "--co_coeff",
    type=float,
    default=1,
    help="tradeoff coefficient for co-labelling loss",
)

# label and unlabel
parser.add_argument(
    "--labeled_bs", type=int, default=12, help="labeled_batch_size per gpu"
)
# parser.add_argument('--labeled_num', type=int, default=136,
parser.add_argument("--labeled_num", type=int, default=7, help="labeled data")
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument(
    "--consistency_type", type=str, default="mse", help="consistency_type"
)
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument(
    "--consistency_rampup", type=float, default=200.0, help="consistency_rampup"
)
args = parser.parse_args()


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "140": 1312,
        }
    elif "Prostate":
        ref_dict = {
            "2": 27,
            "4": 53,
            "8": 120,
            "12": 179,
            "16": 256,
            "21": 312,
            "42": 623,
        }
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    # TODO: make sure that both models have different intialized weights
    model1 = create_model()
    model2 = create_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def get_nl_loss(weak, strong):
        """Get loss for negative learning and related metrics.
        Compares least likely prediction (from strong augment) with argmin of weak augment (comp labels).

        Args:
            weak (batch): weakly augmented batch
            strong (batch): strongly augmented batch

        Returns:
            nl_loss, as_weight, min_preds, comp_labels
        """
        il_output = torch.reshape(
            strong,
            (
                args.batch_size,
                args.num_classes,
                args.patch_size[0] * args.patch_size[1],
            ),
        )
        # calculate entropy for image-level preds (tensor of length labeled_bs)
        as_weight = 1 - (
            Categorical(probs=il_output).entropy()
            / np.log(args.patch_size[0] * args.patch_size[1])
        )
        # batch level average of entropy
        as_weight = torch.mean(as_weight)
        # complementary loss
        comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
        min_preds = torch.add(torch.negative(strong), 1)
        nl_loss = ce_loss(min_preds, comp_labels)
        return nl_loss, as_weight, min_preds, comp_labels

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([WeakStrongAugment(args.patch_size)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print(
        "Total silices is: {}, labeled slices is: {}".format(
            total_slices, labeled_slice
        )
    )
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs
    )

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(
        model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    optimizer2 = optim.SGD(
        model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )

    start_epoch = 0

    # restore previous checkpoint
    if args.load:
        model1, optimizer1, start_epoch1, performance1 = util.restore_model(
            logger=logging, snapshot_path=snapshot_path, model_num="model1_iter"
        )
        model2, optimizer2, start_epoch2, performance2 = util.restore_model(
            logger=logging, snapshot_path=snapshot_path, model_num="model2_iter"
        )
        if start_epoch1 <= start_epoch2:
            start_epoch = start_epoch1
        else:
            start_epoch = start_epoch2

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, weak_batch, strong_batch, label_batch = (
                sampled_batch["image"],
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )

            volume_batch, weak_batch, strong_batch, label_batch = (
                volume_batch.cuda(),
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )

            # unsupervised outputs
            outputs_weak1 = model1(weak_batch)
            outputs_weak_soft1 = torch.softmax(outputs_weak1, dim=1)
            outputs_strong1 = model1(strong_batch)
            outputs_strong_soft1 = torch.softmax(outputs_strong1, dim=1)

            outputs_weak2 = model2(weak_batch)
            outputs_weak_soft2 = torch.softmax(outputs_weak2, dim=1)
            outputs_strong2 = model2(strong_batch)
            outputs_strong_soft2 = torch.softmax(outputs_strong2, dim=1)

            # pseudo labels, masking for threshold
            mask1 = (outputs_weak1 > args.conf_thresh).float()
            outputs_weak_masked1 = torch.softmax(mask1 * outputs_weak1, dim=1)
            pseudo_labels1 = torch.argmax(
                outputs_weak_masked1[args.labeled_bs :].detach(),
                dim=1,
                keepdim=False,
            )

            mask2 = (outputs_weak2 > args.conf_thresh).float()
            outputs_weak_masked2 = torch.softmax(mask2 * outputs_weak2, dim=1)
            pseudo_labels2 = torch.argmax(
                outputs_weak_masked2[args.labeled_bs :].detach(),
                dim=1,
                keepdim=False,
            )

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # LOSSES
            ## Supervised:
            sup_loss1 = ce_loss(
                outputs_weak1[: args.labeled_bs],
                label_batch[:][: args.labeled_bs].long(),
            )
            +dice_loss(
                outputs_weak_soft1[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )

            sup_loss2 = ce_loss(
                outputs_weak2[: args.labeled_bs],
                label_batch[:][: args.labeled_bs].long(),
            )
            +dice_loss(
                outputs_weak_soft2[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )

            ## Unsupervised
            ### Self-labelling losses
            nl_loss1, as_weight1, min_preds1, comp_labels1 = get_nl_loss(
                weak=outputs_weak_soft1,
                strong=outputs_strong_soft1,
            )
            self_loss1 = (
                ce_loss(outputs_strong1[args.labeled_bs :], pseudo_labels1)
                + as_weight1 * nl_loss1
            )

            nl_loss2, as_weight2, min_preds2, comp_labels2 = get_nl_loss(
                weak=outputs_weak_soft2,
                strong=outputs_strong_soft2,
            )
            self_loss2 = (
                ce_loss(outputs_strong2[args.labeled_bs :], pseudo_labels2)
                + as_weight2 * nl_loss2
            )

            ### Co-labelling losses
            co_loss1_mask = (as_weight2 > args.conf_thresh).float()
            co_loss1 = co_loss1_mask * ce_loss(
                outputs_strong1[args.labeled_bs :], pseudo_labels2
            ) + co_loss1_mask * as_weight2 * ce_loss(min_preds1, comp_labels2)

            co_loss2_mask = (as_weight1 > args.conf_thresh).float()
            co_loss2 = co_loss2_mask * ce_loss(
                outputs_strong2[args.labeled_bs :], pseudo_labels1
            ) + co_loss2_mask * as_weight1 * ce_loss(min_preds2, comp_labels1)

            model1_loss = (
                sup_loss1 + args.self_coeff * self_loss1 + args.co_coeff * co_loss1
            )
            model2_loss = (
                sup_loss2 + args.self_coeff * self_loss2 + args.co_coeff * co_loss2
            )
            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group["lr"] = lr_
            for param_group in optimizer2.param_groups:
                param_group["lr"] = lr_

            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar(
                "weights/consistency_weight", consistency_weight, iter_num
            )
            writer.add_scalar(
                "weights/model1_adaptive_sample_weight", as_weight1, iter_num
            )
            writer.add_scalar(
                "weights/model2_adaptive_sample_weight", as_weight2, iter_num
            )
            writer.add_scalar("loss/model1_loss", model1_loss, iter_num)
            writer.add_scalar("loss/model2_loss", model2_loss, iter_num)
            logging.info(
                "iteration %d : model1 loss : %f model2 loss : %f"
                % (iter_num, model1_loss.item(), model2_loss.item())
            )
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image("train/Image", image, iter_num)
                outputs = torch.argmax(
                    torch.softmax(outputs_weak1, dim=1), dim=1, keepdim=True
                )
                writer.add_image(
                    "train/model1_Prediction", outputs[1, ...] * 50, iter_num
                )
                outputs = torch.argmax(
                    torch.softmax(outputs_weak2, dim=1), dim=1, keepdim=True
                )
                writer.add_image(
                    "train/model2_Prediction", outputs[1, ...] * 50, iter_num
                )
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model1,
                        classes=num_classes,
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/model1_val_{}_dice".format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num,
                    )
                    writer.add_scalar(
                        "info/model1_val_{}_hd95".format(class_i + 1),
                        metric_list[class_i, 1],
                        iter_num,
                    )

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar("info/model1_val_mean_dice", performance1, iter_num)
                writer.add_scalar("info/model1_val_mean_hd95", mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model1_iter_{}_dice_{}.pth".format(
                            iter_num, round(best_performance1, 4)
                        ),
                    )
                    save_best = os.path.join(
                        snapshot_path, "{}_best_model1.pth".format(args.model)
                    )
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    "iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f"
                    % (iter_num, performance1, mean_hd951)
                )
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model2,
                        classes=num_classes,
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/model2_val_{}_dice".format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num,
                    )
                    writer.add_scalar(
                        "info/model2_val_{}_hd95".format(class_i + 1),
                        metric_list[class_i, 1],
                        iter_num,
                    )

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar("info/model2_val_mean_dice", performance2, iter_num)
                writer.add_scalar("info/model2_val_mean_hd95", mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model2_iter_{}_dice_{}.pth".format(
                            iter_num, round(best_performance2)
                        ),
                    )
                    save_best = os.path.join(
                        snapshot_path, "{}_best_model2.pth".format(args.model)
                    )
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    "iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f"
                    % (iter_num, performance2, mean_hd952)
                )
                model2.train()

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer1.param_groups:
                    param_group["lr"] = lr_
                for param_group in optimizer2.param_groups:
                    param_group["lr"] = lr_
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "model1_iter_" + str(iter_num) + ".pth"
                )
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, "model2_iter_" + str(iter_num) + ".pth"
                )
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


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

    snapshot_path = "../model/{}_{}/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")
    shutil.copytree(
        ".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"])
    )

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
