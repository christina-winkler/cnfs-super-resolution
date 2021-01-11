import torch
import torch.nn as nn
import numpy as np
import random

from models.transformations import modules
from models.architectures import RRDBNet_arch as arch

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LrNet(nn.Module):
    def __init__(self, in_c, cond_channels, s, input_shape, nb, gc=32):
        """
        Args:
            in_c (int): Number of input channels.
            cond_channels (int): Channel size of the feature map the
            network outputs.
            nb (int): Nr of residual blocks to build the architecture.
        Returns:
            out (tensor): Feature map computed by Residual-in-Residual
                          dense block network architecture
                          [bsz, cond_channels, height, width].
        """
        super().__init__()
        (c, w, h) = input_shape
        self.RRDBNet = arch.RRDBNet(in_c, cond_channels, nb, s, input_shape, gc=gc)

    def forward(self, x):
        out = self.RRDBNet(x)
        return out


class FlowStep(nn.Module):
    def __init__(
        self,
        level,
        s,
        channel_dim,
        input_shape,
        filter_size,
        cond_channels,
        noscale,
        noscaletest,
    ):
        super().__init__()
        # 1. Activation Normalization
        self.actnorm = modules.ActNorm(channel_dim)
        # 2. Invertible 1x1 Convolution
        self.invconv = modules.Invert1x1Conv(channel_dim)
        # 3. Conditional Coupling layer
        self.conditionalCoupling = modules.ConditionalCoupling(
            level,
            s,
            channel_dim,
            input_shape,
            cond_channels,
            filter_size,
            noscale,
            noscaletest,
        )

    def forward(self, z, lr_feat_map=None, x_lr=None, logdet=0, reverse=False):

        if not reverse:
            # 1. Activation normalization layer
            (
                z,
                logdet,
            ) = self.actnorm(z, logdet=logdet, reverse=False)
            # 2. Permutation with invertible 1x1 Convolutional layer
            z, logdet = self.invconv.forward_(z, logdet=logdet, reverse=False)
            # 3. Conditional Coupling Operation
            z, logdet = self.conditionalCoupling(
                z, lr_feat_map=lr_feat_map, logdet=logdet, reverse=False
            )

            return z, logdet

        else:
            # 1. Conditional Coupling
            z, logdet = self.conditionalCoupling(
                z, lr_feat_map=lr_feat_map, logdet=logdet, reverse=True
            )
            # 2. Invertible 1x1 convolution
            z, logdet = self.invconv.forward_(z, logdet=logdet, reverse=True)
            # 3. Actnorm
            z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

            return z, logdet


class NormFlowNet(nn.Module):
    def __init__(
        self,
        input_shape,
        filter_size,
        bsz,
        s,
        L,
        K,
        nb,
        cond_channels,
        noscale,
        noscaletest,
    ):

        super().__init__()
        self.L = L
        self.K = K
        self.bsz = bsz
        C, H, W = input_shape
        self.output_shapes = []
        self.layers = nn.ModuleList()
        self.lrNet = LrNet(
            in_c=3,
            cond_channels=cond_channels,
            s=s,
            input_shape=(C, W // s, H // s),
            nb=nb,
        )
        self.downsample_convs = nn.ModuleList()

        for i in range(self.L):
            self.downsample_convs.append(
                nn.Conv2d(cond_channels, cond_channels, 2, stride=2, padding=0)
            )

        # Build Normalizing Flow
        self.level_modules = torch.nn.ModuleList()
        for i in range(self.L):
            self.level_modules.append(nn.ModuleList())

        for i in range(self.L):
            # 1. Squeeze Layer
            self.level_modules[i].append(modules.Squeeze(factor=2))
            C, H, W = C * 4, H // 2, W // 2
            self.output_shapes.append((-1, C, H, W))

            # 2. Flow Steps
            for k in range(K):
                self.level_modules[i].append(
                    FlowStep(
                        i,
                        s,
                        C,
                        input_shape,
                        filter_size,
                        cond_channels,
                        noscale,
                        noscaletest,
                    )
                )

            if i < L - 1:
                # 3.Split Prior for intermediate latent variables
                self.level_modules[i].append(
                    modules.GaussianPrior(C, s, cond_channels, (bsz, C, H, W))
                )
                C = C // 2
                self.output_shapes.append((-1, C, H, W))

        self.level_modules[-1].append(
            modules.GaussianPrior(C, s, cond_channels, (bsz, C, H, W), final=True)
        )

    def forward(
        self, z, xlr=None, logdet=0, logpz=0, eps=None, reverse=False, use_stored=False
    ):

        # Pre-compute LR feature map
        lr_feat_map = self.lrNet(xlr)
        lr_downsampled_feats = [lr_feat_map]
        for i in range(self.L):
            lr_downsampled_feats.append(
                self.downsample_convs[i](lr_downsampled_feats[-1])
            )

        # Encode
        if not reverse:
            for i in range(self.L):
                for layer in self.level_modules[i]:
                    if isinstance(layer, modules.Squeeze):
                        z = layer(z, reverse=False)
                    elif isinstance(layer, FlowStep):
                        z, logdet = layer(
                            z,
                            lr_feat_map=lr_downsampled_feats[i + 1],
                            x_lr=xlr,
                            logdet=logdet,
                            reverse=False,
                        )
                    elif isinstance(layer, modules.GaussianPrior):
                        z, logdet, logpz = layer(
                            z,
                            logdet=logdet,
                            logpz=logpz,
                            lr_feat_map=lr_downsampled_feats[i + 1],
                            eps=eps,
                            reverse=False,
                        )
        else:
            # Decode
            for i in reversed(range(self.L)):
                for layer in reversed(self.level_modules[i]):
                    if isinstance(layer, modules.GaussianPrior):
                        z, logdet, logpz = layer(
                            z,
                            lr_feat_map=lr_downsampled_feats[i + 1],
                            logdet=logdet,
                            logpz=logpz,
                            eps=eps,
                            reverse=True,
                            use_stored=use_stored,
                        )
                    elif isinstance(layer, FlowStep):
                        z, logdet = layer(
                            z=z,
                            lr_feat_map=lr_downsampled_feats[i + 1],
                            x_lr=xlr,
                            logdet=logdet,
                            reverse=True,
                        )
                    elif isinstance(layer, modules.Squeeze):
                        z = layer(z, reverse=True)

        return z, logdet, logpz


class FlowModel(nn.Module):
    def __init__(
        self,
        input_shape,
        filter_size,
        L,
        K,
        bsz,
        s,
        nb,
        cond_channels=128,
        n_bits_x=8,
        noscale=False,
        noscaletest=False,
    ):

        super().__init__()

        self.flow = NormFlowNet(
            input_shape=input_shape,
            filter_size=filter_size,
            s=s,
            bsz=bsz,
            K=K,
            L=L,
            nb=nb,
            cond_channels=cond_channels,
            noscale=noscale,
            noscaletest=noscaletest,
        )

        self._variational_dequantizer = None
        self.nbins = 2 ** n_bits_x

    def forward(
        self,
        x_hr=None,
        xlr=None,
        z=None,
        logdet=0,
        eps=None,
        reverse=False,
        use_stored=False,
    ):

        if not reverse:
            return self.normalizing_flow(x_hr, xlr)

        else:
            return self.inverse_flow(
                z=z, x=xlr, logdet=logdet, eps=eps, use_stored=use_stored
            )

    def normalizing_flow(self, x_hr, x_lr):

        # Dequantize pixels: Discrete -> Continuous
        z, logdet = self._dequantize_uniform(x_hr, self.nbins)

        # Push z through flow
        z, logdet, logp_z = self.flow.forward(z=z, xlr=x_lr, logdet=logdet)

        # Loss: Z'ks under Gaussian + sum_logdet
        D = float(np.log(2) * np.prod(x_hr.size()[1:]))
        x_bpd = -(logdet + logp_z) / D
        # loss = x_bpd + 0.001 * l2_scale
        return z, x_bpd

    def inverse_flow(self, z, xlr, eps, logdet=0, use_stored=False):
        x = self.flow.forward(
            z, logdet=logdet, xlr=xlr, eps=eps, reverse=True, use_stored=use_stored
        )
        return x

    def _dequantize_uniform(self, x, n_bins):
        """
        Rescales pixels and adds uniform noise for dequantization.
        """
        unif_noise = torch.zeros_like(x).uniform_(0, float(1.0 / n_bins))
        x = unif_noise + x

        # Initialize log determinant
        logdet = torch.zeros_like(x[:, 0, 0, 0])

        # Log determinant adjusting for rescaling of 1/nbins for each pixel value
        logdet += float(-np.log(n_bins) * np.prod(x.size()[1:]))
        return x, logdet

    def _sample(self, x, eps=None):
        """
        Super-resolves a low-resolution image with estimated params.
        """
        # Draw samples from model
        with torch.no_grad():
            samples = self.inverse_flow(z=None, xlr=x, eps=eps)[0]
            return samples.clamp(min=0, max=float(self.nbins - 1) / float(self.nbins))
