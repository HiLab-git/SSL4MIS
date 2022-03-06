import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from random import uniform


class Rotated_MIP_Loss_Multiclass(nn.Module):
    # 监督为softmax后的输出
    def __init__(self, num_rot=3,  device='cuda'):
        super(Rotated_MIP_Loss_Multiclass, self).__init__()
        self.device = device
        assert isinstance(num_rot, int) and num_rot > 0
        self.num_rot = num_rot

    def forward(self, batch_input, label_input):
        rot_list = [uniform(0, 1.57) for _ in range(self.num_rot)]

        size = batch_input.shape
        loss = self.max_project_loss(batch_input, label_input)
        # print(loss)
        for i in rot_list:
            rot_mat = self.create_rot_matrix(size, i)
            rot_grid = F.affine_grid(rot_mat, size).to(self.device)
            inputs = F.grid_sample(batch_input, rot_grid)
            labels = F.grid_sample(label_input, rot_grid)
            loss_ = self.max_project_loss(inputs, labels)
            loss += self.max_project_loss(inputs, labels)
            # print(loss_)
        return loss / (len(rot_list) + 1)

    def create_rot_matrix(self, size, angle):
        b = size[0]
        angle = torch.tensor(angle).to(self.device)
        rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
        rotation_matrix[:, 0, 0] = torch.cos(angle)
        rotation_matrix[:, 0, 1] = -torch.sin(angle)
        rotation_matrix[:, 1, 0] = torch.sin(angle)
        rotation_matrix[:, 1, 1] = torch.cos(angle)
        rotation_matrix[:, 2, 2] = 1.0
        return rotation_matrix[:, :2, :]

    def max_project_loss(self, score, target):
        total_loss = 0.0
        for index, i in enumerate([-1, -2]):
            new_target = torch.max(target, dim=i)[0].float()
            new_score = torch.max(score, dim=i)[0].float()
            smooth = 1e-5
            intersect = torch.sum(new_score * new_target, dim=-1)
            y_sum = torch.sum(new_target, dim=-1)
            z_sum = torch.sum(new_score, dim=-1)
            loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
            loss = torch.mean(loss[:, 1:])   # 这里去掉了背景
            total_loss += (1.0 - loss)
        return total_loss / 2

    def mean_project_loss(self, score, target):
        total_loss = 0.0
        for index, i in enumerate([-1, -2]):
            new_target = torch.mean(target, dim=i, keepdim=True).float()
            new_score = torch.mean(score, dim=i, keepdim=True).float()
            new_target = new_target / new_target.max()
            new_score = new_score / new_score.max()
            loss = torch.nn.functional.mse_loss(new_score, new_target)
            total_loss += loss
        return total_loss / 2


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib

    xs = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]) /10.
    ys = [0.035633184015750885, 0.04039973020553589, 0.04840249940752983, 0.049674104899168015, 0.05561297759413719, 0.06043963134288788, 0.06371262669563293, 0.06686598062515259, 0.07038842141628265, 0.07373663783073425, 0.07658594101667404, 0.07925296574831009, 0.08191808313131332, 0.08441270142793655, 0.08767103403806686, 0.09013877063989639, 0.09172293543815613, 0.09327001124620438, 0.09405311197042465, 0.0982089713215828, 0.0963851660490036, 0.10003411769866943, 0.10333031415939331, 0.10372356325387955, 0.10573377460241318, 0.1040252298116684, 0.10658720880746841, 0.10942418873310089, 0.10836256295442581, 0.1106153279542923, 0.11024092882871628, 0.11183327436447144, 0.11297039687633514, 0.11287706345319748, 0.11342803388834, 0.11346597969532013, 0.11305929720401764, 0.11496424674987793, 0.11747633665800095, 0.1148373931646347]

    zs = [0.0526] * 40
    ms = [0.0156] * 40

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 16})
    # plt.rc('axes', titlesize=20)
    plt.figure()
    plt.plot(xs, ys, label='MPR loss')
    plt.plot(xs, zs, label='Dice loss')
    plt.plot(xs, ms, label='MSE loss')
    # plt.xlabel("Position of disagreement region", fontsize=15)
    # plt.ylabel("Loss value", fontsize=15)
    plt.legend(fontsize=16)
    plt.show()
    #
    # rmip = Rotated_MIP_Loss_Multiclass(num_rot=360, device='cpu')
    #
    #
    # xs = []
    # mprs = []
    # for i in range(40):
    #     pred_a = torch.zeros((800, 800))
    #     pred_b = torch.zeros((800, 800))
    #
    #     pred_a[200: 500, 0: 300] = 1
    #     # pred_a[300: 400, 400: 500] = 1
    #
    #     pi = i * 10
    #     pred_a[300: 400, 300+pi: 400+pi] = 1
    #
    #     pred_b[200: 500, 0: 300] = 1
    #
    #     pred_a = pred_a.unsqueeze(dim=0).unsqueeze(dim=0)
    #     pred_b = pred_b.unsqueeze(dim=0).unsqueeze(dim=0)
    #
    #
    #     print("dice: ", 1- 2 * (pred_a * pred_b).sum() / (pred_a.sum() + pred_b.sum()))
    #
    #     print("mse: ", ((pred_a - pred_b) ** 2).mean())
    #
    #     mpr = rmip(pred_a, pred_b).item()
    #     print("mpr mean: ", mpr)
    #
    #     xs.append(i)
    #     mprs.append(mpr)
    #
    # print(xs)
    # print(mprs)
