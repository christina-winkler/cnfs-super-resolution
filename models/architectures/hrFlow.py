import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import utils
import glow

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import modules


class FlowStep(nn.Module):
    def __init__(self, channel_dim, filter_size, args):
        super().__init__()
        # 1. Activation Normalization
        self.actnorm = modules.ActNorm(channel_dim, args=args)
        # 2. Invertible 1x1 Convolution
        self.invconv = modules.Invert1x1Conv(channel_dim)
        # 3. Affine Coupling layer
        self.affineCoupling = modules.AffineCoupling(
            channel_dim, filter_size, args=args
        )

    def forward(self, z, logdet=0, reverse=False):
        if not reverse:
            # 1. Activation normalization layer
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)
            # 2. Permutation with invertible 1x1 Convolutional layer
            z, logdet = self.invconv.forward_(z, logdet=logdet, reverse=False)
            # 3. Affine Coupling Operation
            z, logdet = self.affineCoupling(z, logdet)

            return z, logdet

        else:
            # 1. Affine Coupling
            z, logdet = self.affineCoupling(z, logdet, reverse=True)
            # 2. Invertible 1x1 convolution
            z, logdet = self.invconv.forward_(z, logdet, reverse=True)
            # 3. Actnorm
            z, logdet = self.actnorm(z, logdet, reverse=True)

            return z, logdet


class HRNormFlowNet(nn.Module):
    def __init__(self, input_shape, filter_size, bsz, L, K, args):

        super().__init__()
        self.L = L
        self.K = K
        C, H, W = input_shape
        self.layers = nn.ModuleList()
        self.output_shapes = []

        # Build Normalizing Flow
        for i in range(self.L):

            # 1. Squeeze Layer
            self.layers.append(modules.Squeeze(factor=2))
            C, H, W = C * 4, H // 2, W // 2
            self.output_shapes.append((-1, C, H, W))

            # 2. Flow Steps
            for k in range(K):
                self.layers.append(FlowStep(C, filter_size, args))

            # 3.Split Prior for intermediate latent variables
            self.layers.append(modules.Split(C))
            C = C // 2
            self.output_shapes.append((-1, C, H, W))

        # self.layers.append(modules.GaussianPrior((bsz, C, H, W)))

    def forward(self, z, eps=0, logdet=0, logpz=0, reverse=False, use_stored=False):

        # Encode
        if not reverse:
            for layer in self.layers:
                if isinstance(layer, modules.Split):
                    z, logdet, logpz = layer(
                        z,
                        eps=eps,
                        logdet=logdet,
                        logpz=logpz,
                        reverse=False,
                        use_stored=use_stored,
                    )
                elif isinstance(layer, FlowStep):
                    z, logdet = layer(z, logdet=logdet, reverse=False)
                elif isinstance(layer, modules.Squeeze):
                    z = layer(z, reverse=False)

        else:
            # Decode
            for layer in reversed(self.layers):
                if isinstance(layer, modules.Split):
                    z, logdet, logpz = layer(
                        z,
                        eps=eps,
                        logdet=logdet,
                        logpz=logpz,
                        reverse=True,
                        use_stored=use_stored,
                    )
                elif isinstance(layer, FlowStep):
                    z, logdet = layer(z, logdet=logdet, reverse=True)
                elif isinstance(layer, modules.Squeeze):
                    z = layer(z, reverse=True)

        return z, logdet, logpz


class LRNormFlowNet(nn.Module):
    def __init__(self, input_shape, filter_size, bsz, L, K, args):

        super().__init__()
        self.L = L
        self.K = K
        C, H, W = input_shape
        self.layers = nn.ModuleList()
        self.output_shapes = []

        # Build Normalizing Flow
        for i in range(self.L):

            # 1. Squeeze Layer
            self.layers.append(modules.Squeeze(factor=2))
            C, H, W = C * 4, H // 2, W // 2
            self.output_shapes.append((-1, C, H, W))

            # 2. Flow Steps
            for k in range(K):
                self.layers.append(FlowStep(C, filter_size, args))

            if i < L - 1:
                # 3.Split Prior for intermediate latent variables
                self.layers.append(modules.Split(C))
                C = C // 2
                self.output_shapes.append((-1, C, H, W))

        self.layers.append(modules.GaussianPrior((bsz, C, H, W)))

    def forward(self, z, eps=0, logdet=0, logpz=0, reverse=False, use_stored=False):

        # Encode
        if not reverse:
            for layer in self.layers:
                if isinstance(layer, modules.Split):
                    z, logdet, logpz = layer(
                        z,
                        eps=eps,
                        logdet=logdet,
                        logpz=logpz,
                        reverse=False,
                        use_stored=use_stored,
                    )
                elif isinstance(layer, FlowStep):
                    z, logdet = layer(z, logdet=logdet, reverse=False)
                elif isinstance(layer, modules.Squeeze):
                    z = layer(z, reverse=False)
                elif isinstance(layer, modules.GaussianPrior):
                    z, logdet, logpz = layer(
                        x=z, eps=eps, logdet=logdet, logpz=logpz, reverse=False
                    )

        else:
            # Decode
            for layer in reversed(self.layers):
                if isinstance(layer, modules.Split):
                    z, logdet, logpz = layer(
                        z,
                        eps=eps,
                        logdet=logdet,
                        logpz=logpz,
                        reverse=True,
                        use_stored=use_stored,
                    )
                elif isinstance(layer, FlowStep):
                    z, logdet = layer(z, logdet=logdet, reverse=True)
                elif isinstance(layer, modules.Squeeze):
                    z = layer(z, reverse=True)
                elif isinstance(layer, modules.GaussianPrior):
                    z, logdet, logpz = layer(
                        x=z, eps=eps, logdet=logdet, logpz=logpz, reverse=True
                    )

        return z, logdet, logpz


class FlowModel(nn.Module):
    def __init__(self, flownet, n_bits_x=8):

        super().__init__()
        self.flow = flownet
        self.n_bins = 2 ** n_bits_x

    def forward(
        self,
        y=None,
        z=None,
        eps=0,
        logdet=0,
        reverse=False,
        use_stored=False,
        dequantize=True,
    ):

        if not reverse:
            return self.normalizing_flow(
                z=y, eps=eps, use_stored=use_stored, dequantize=dequantize
            )

        else:
            return self.inverse_flow(z=z, eps=eps, logdet=logdet, use_stored=use_stored)

    def normalizing_flow(self, z=None, eps=0, use_stored=False, dequantize=True):

        # Dequantize pixels: Discrete -> Continuous
        if dequantize:
            z, logdet = self._dequantize(z, self.n_bins)
        else:
            logdet = torch.zeros_like(z[:, 0, 0, 0])

        # Push z through flow
        z, logdet, logp_z = self.flow.forward(
            z=z, eps=eps, logdet=logdet, use_stored=use_stored
        )

        # Loss: Z'ks under Gaussian + sum_logdet
        x_bpd = -(logdet + logp_z) / float(np.log(2) * np.prod(z.size()[1:]))

        return z, x_bpd, logdet, logp_z

    def inverse_flow(self, z, eps=0, logdet=0, use_stored=False):
        x, _, _ = self.flow(
            z=z, eps=eps, logdet=logdet, reverse=True, use_stored=use_stored
        )
        return x

    def _dequantize(self, x, n_bins):
        """
        Rescales pixels and adds uniform noise to pixels to dequantize them.
        """

        # Add uniform noise to each pixel to transform them from discrete to continuous values
        unif_noise = torch.zeros_like(x).uniform_(0, 1.0 / n_bins)
        x = unif_noise + x

        # Initialize log determinant
        logdet = torch.zeros_like(x[:, 0, 0, 0])

        # Log determinant adjusting for rescaling of 1/256 for each pixel value
        logdet += -np.log(n_bins) * x.size(1) * x.size(2) * x.size(3)
        return x, logdet

    def _sample_images(self):
        with torch.no_grad():
            samples = self.inverse_flow(z=None)[0]
        return samples


class HierarchicalFlow(nn.Module):
    def __init__(self, args, n_bits_x=8):

        super().__init__()

        C, W, H = args.im_shape_hr

        main_flow_s2 = HRNormFlowNet(
            args.im_shape_hr, args.filter_size, args.bsz, args.hrL, args.hrK, args=args
        )
        main_flow_s4 = HRNormFlowNet(
            args.im_shape_hr, args.filter_size, args.bsz, args.hrL, args.hrK, args=args
        )

        side_flow2 = LRNormFlowNet(
            (C, W // 2, H // 2),
            args.filter_size,
            args.bsz,
            args.lrL,
            args.lrK,
            args=args,
        )

        side_flow4 = LRNormFlowNet(
            (C, W // 2, H // 2),
            args.filter_size,
            args.bsz,
            args.lrL,
            args.lrK,
            args=args,
        )

        self.main_flow_s2 = FlowModel(main_flow_s2, n_bits_x)
        self.main_flow_s4 = FlowModel(main_flow_s4, n_bits_x)

        self.flow_s2 = FlowModel(side_flow2, n_bits_x)
        self.flow_s4 = FlowModel(side_flow4, n_bits_x)

        self.squeezer = modules.Squeeze()
        self.bsz = args.bsz

    # self.wasserstein_dist = SamplesLoss(loss="sinkhorn",p=2,blur=0.5,scaling=.8)

    def forward(self, y=None, x2=None, x4=None):
        # ------------------Scale 1 ------------------------#
        # Scale 1, image is downsampled by scale factor 2
        u1, _, logdeteps1, logpeps1 = self.main_flow_s2(y=y)
        z1x, bpd1x, logdetz1x, logpz1x = self.flow_s2(y=x2)
        x_hat1 = self.squeezer(u1, reverse=True)
        zxhat1, bpdxhat1, logdetzxhat1, logpzxhat1 = self.flow_s2(
            y=x_hat1, dequantize=False
        )

        # L2 norm on z, zhat
        diffz1 = (
            zxhat1.view(self.bsz, -1).contiguous() - z1x.view(self.bsz, -1).contiguous()
        )
        l2_normz1 = torch.norm(diffz1, p=2, dim=1)

        # L2 norm on downsampled x
        diffx1 = (
            x_hat1.view(self.bsz, -1).contiguous() - x2.view(self.bsz, -1).contiguous()
        )

        l2_normx1 = torch.norm(diffx1, p=2, dim=1)

        # ------------------Scale 2 --------------------------#

        u2, _, logdet_eps2, logp_eps2 = self.main_flow_s4(y=x2)
        z2x, bpd2x, logdetz2x, logpz2x = self.flow_s4(y=x4)
        x_hat2 = self.squeezer(u2, reverse=True)
        zxhat2, bpdxhat2, logdetzxhat2, logpzxhat2 = self.flow_s4(
            y=x_hat2, dequantize=False
        )

        # L2 norm on z, zhat
        diffz2 = (
            zxhat2.view(self.bsz, -1).contiguous() - z2x.view(self.bsz, -1).contiguous()
        )
        l2_normz2 = torch.norm(diffz2, p=2, dim=1)

        # L2 norm on downsampled x
        diffx2 = (
            x_hat2.view(self.bsz, -1).contiguous() - x4.view(self.bsz, -1).contiguous()
        )

        l2_normx2 = torch.norm(diffx2, p=2, dim=1)

        # ------------------- BPD(y) ---------------------------#

        # compute bpd of y
        # nll_y = logp_eps + logdet_eps + logpzx +
        terms_epsilon = logpeps1 + logdeteps1 + logp_eps2 + logdet_eps2
        nll_y = terms_epsilon + logdetz1x + logpz1x + logpz2x + logdetz2x
        y_dims = float(np.log(2) * np.prod(y.size()[1:]))
        bpd_y = -(nll_y) / y_dims

        loss = (
            bpd_y
            + 0.1 * l2_normx2
            + 0.1 * l2_normx1
            + 0.1 * l2_normz1
            + 0.1 * l2_normz2
        )

        return z1x, bpd_y, loss

    def _upsample2(self, x, eps=0.6):
        with torch.no_grad():
            x_ = self.squeezer(x)
            y_hat = self.main_flow_s2(
                z=x_, eps=eps, reverse=True, use_stored=False, dequantize=True
            )
        return y_hat

    def _upsample4(self, x, eps=0.6):
        with torch.no_grad():
            x_ = self.squeezer(x)
            y_hat = self.main_flow_s4(
                z=x_, eps=eps, reverse=True, use_stored=False, dequantize=True
            )
        return y_hat

    def _reconstruct_y(self, u):
        with torch.no_grad():
            y = self.hr_flow(z=u, reverse=True, use_stored=True)
        return y

    def _reconstruct_x(self, zx):
        with torch.no_grad():
            x = self.lr_flow(z=zx, reverse=True, use_stored=True)
        return x
