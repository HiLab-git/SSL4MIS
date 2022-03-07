import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=[10, 1, 1])
        asd = metric.binary.asd(pred, gt, voxelspacing=[10, 1, 1])
        return dice, hd95, asd
    else:
        return 0, 50, 10


def test_single_volume(image, label, net, classes, patch_size=[256, 256], batch_size=8):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        ind_x = np.array([i for i in range(image.shape[0])])
        for ind in ind_x[::batch_size]:
            if ind + batch_size < image.shape[0]:
                slice = image[ind:ind + batch_size, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(
                    slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        net(input), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[ind:ind + batch_size, ...] = pred
            else:
                slice = image[ind:, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(
                    slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(
                        net(input), dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[ind:, ...] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


# def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
#     image, label = image.squeeze(0).cpu().detach(
#     ).numpy(), label.squeeze(0).cpu().detach().numpy()
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         ind_x = np.array([i for i in range(image.shape[0])])

#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             slice = zoom(
#                 slice, (patch_size[0] / x, patch_size[1] / y), order=0)
#             input = torch.from_numpy(slice).unsqueeze(
#                 0).unsqueeze(0).float().cuda()
#             net.eval()
#             with torch.no_grad():
#                 out = torch.argmax(torch.softmax(
#                     net(input), dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
#                 pred = zoom(
#                     out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(
#                 net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(
#             prediction == i, label == i))

#     # return metric_list, image, prediction, label
#     return metric_list

#
# def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
#     image, label = image.squeeze(0).cpu().detach(
#     ).numpy(), label.squeeze(0).cpu().detach().numpy()
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             slice = zoom(
#                 slice, (patch_size[0] / x, patch_size[1] / y), order=0)
#             input = torch.from_numpy(slice).unsqueeze(
#                 0).unsqueeze(0).float().cuda()
#             net.eval()
#             with torch.no_grad():
#                 output_main, _, _, _ = net(input)
#                 out = torch.argmax(torch.softmax(
#                     output_main, dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
#                 pred = zoom(
#                     out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             output_main, _, _, _ = net(input)
#             out = torch.argmax(torch.softmax(
#                 output_main, dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(
#             prediction == i, label == i))
#     return metric_list
#
#


def test_single_volume_multitask(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    # return metric_list, image, prediction, label
    return metric_list
