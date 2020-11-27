import torch
from torch import nn
from torch.nn import functional as F
from networks.networks_other import init_weights


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')


    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )


class GridAttentionBlock3D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )

class _GridAttentionBlockND_TORR(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(1,1,1), bn_layer=True, use_W=True, use_phi=True, use_theta=True, use_psi=True, nonlinearity1='relu'):
        super(_GridAttentionBlockND_TORR, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_softmax',
                        'concatenation_sigmoid', 'concatenation_mean',
                        'concatenation_range_normalise', 'concatenation_mean_flow']

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_factor = sub_sample_factor if isinstance(sub_sample_factor, tuple) else tuple([sub_sample_factor])*dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # initialise id functions
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.W = lambda x: x
        self.theta = lambda x: x
        self.psi = lambda x: x
        self.phi = lambda x: x
        self.nl1 = lambda x: x

        if use_W:
            if bn_layer:
                self.W = nn.Sequential(
                    conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                    bn(self.in_channels),
                )
            else:
                self.W = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)

        if use_theta:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)


        if use_phi:
            self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                               kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)


        if use_psi:
            self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)


        if nonlinearity1:
            if nonlinearity1 == 'relu':
                self.nl1 = lambda x: F.relu(x, inplace=True)

        if 'concatenation' in mode:
            self.operation_function = self._concatenation
        else:
            raise NotImplementedError('Unknown operation function.')

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')


        if use_psi and self.mode == 'concatenation_sigmoid':
            nn.init.constant(self.psi.bias.data, 3.0)

        if use_psi and self.mode == 'concatenation_softmax':
            nn.init.constant(self.psi.bias.data, 10.0)

        # if use_psi and self.mode == 'concatenation_mean':
        #     nn.init.constant(self.psi.bias.data, 3.0)

        # if use_psi and self.mode == 'concatenation_range_normalise':
        #     nn.init.constant(self.psi.bias.data, 3.0)

        parallel = False
        if parallel:
            if use_W: self.W = nn.DataParallel(self.W)
            if use_phi: self.phi = nn.DataParallel(self.phi)
            if use_psi: self.psi = nn.DataParallel(self.psi)
            if use_theta: self.theta = nn.DataParallel(self.theta)

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        #############################
        # compute compatibility score

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w)
        # phi   => (b, c, t, h, w) -> (b, i_c, t, h, w)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        #  nl(theta.x + phi.g + bias) -> f = (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)

        f = theta_x + phi_g
        f = self.nl1(f)

        psi_f = self.psi(f)

        ############################################
        # normalisation -- scale compatibility score
        #  psi^T . f -> (b, 1, t/s1, h/s2, w/s3)
        if self.mode == 'concatenation_softmax':
            sigm_psi_f = F.softmax(psi_f.view(batch_size, 1, -1), dim=2)
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        elif self.mode == 'concatenation_mean':
            psi_f_flat = psi_f.view(batch_size, 1, -1)
            psi_f_sum = torch.sum(psi_f_flat, dim=2)#clamp(1e-6)
            psi_f_sum = psi_f_sum[:,:,None].expand_as(psi_f_flat)

            sigm_psi_f = psi_f_flat / psi_f_sum
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        elif self.mode == 'concatenation_mean_flow':
            psi_f_flat = psi_f.view(batch_size, 1, -1)
            ss = psi_f_flat.shape
            psi_f_min = psi_f_flat.min(dim=2)[0].view(ss[0],ss[1],1)
            psi_f_flat = psi_f_flat - psi_f_min
            psi_f_sum = torch.sum(psi_f_flat, dim=2).view(ss[0],ss[1],1).expand_as(psi_f_flat)

            sigm_psi_f = psi_f_flat / psi_f_sum
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        elif self.mode == 'concatenation_range_normalise':
            psi_f_flat = psi_f.view(batch_size, 1, -1)
            ss = psi_f_flat.shape
            psi_f_max = torch.max(psi_f_flat, dim=2)[0].view(ss[0], ss[1], 1)
            psi_f_min = torch.min(psi_f_flat, dim=2)[0].view(ss[0], ss[1], 1)

            sigm_psi_f = (psi_f_flat - psi_f_min) / (psi_f_max - psi_f_min).expand_as(psi_f_flat)
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])

        elif self.mode == 'concatenation_sigmoid':
            sigm_psi_f = F.sigmoid(psi_f)
        else:
            raise NotImplementedError

        # sigm_psi_f is attention map! upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D_TORR(_GridAttentionBlockND_TORR):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(1,1), bn_layer=True,
                 use_W=True, use_phi=True, use_theta=True, use_psi=True,
                 nonlinearity1='relu'):
        super(GridAttentionBlock2D_TORR, self).__init__(in_channels,
                                               inter_channels=inter_channels,
                                               gating_channels=gating_channels,
                                               dimension=2, mode=mode,
                                               sub_sample_factor=sub_sample_factor,
                                               bn_layer=bn_layer,
                                               use_W=use_W,
                                               use_phi=use_phi,
                                               use_theta=use_theta,
                                               use_psi=use_psi,
                                               nonlinearity1=nonlinearity1)


class GridAttentionBlock3D_TORR(_GridAttentionBlockND_TORR):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(1,1,1), bn_layer=True):
        super(GridAttentionBlock3D_TORR, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   bn_layer=bn_layer)



if __name__ == '__main__':
    from torch.autograd import Variable

    mode_list = ['concatenation']

    for mode in mode_list:

        img = Variable(torch.rand(2, 16, 10, 10, 10))
        gat = Variable(torch.rand(2, 64, 4, 4, 4))
        net = GridAttentionBlock3D(in_channels=16, inter_channels=16, gating_channels=64, mode=mode, sub_sample_factor=(2,2,2))
        out, sigma = net(img, gat)
        print(out.size())
