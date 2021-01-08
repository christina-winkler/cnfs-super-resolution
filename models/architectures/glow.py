import torch
import torch.nn as nn
import numpy as np

import glow_modules as modules


class FlowStep(nn.Module):
    def __init__(self, channel_dim, filter_size):
        super().__init__()
        # 1. Activation Normalization
        self.actnorm = modules.ActNorm(channel_dim)
        # 2. Invertible 1x1 Convolution
        self.invconv = modules.Invert1x1Conv(channel_dim)
        # 3. Affine Coupling layer
        self.affineCoupling = modules.AffineCoupling(channel_dim, filter_size)

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


class NormFlowNet(nn.Module):
    def __init__(self, input_shape, filter_size, bsz, L, K):

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
                self.layers.append(FlowStep(C, filter_size))

            if i < L - 1:
                # 3.Split Prior for intermediate latent variables
                self.layers.append(modules.Split(C))
                C = C // 2
                self.output_shapes.append((-1, C, H, W))

        self.layers.append(modules.GaussianPrior((bsz, C, H, W)))

    def forward(self, z, logdet=0, logpz=0, eps=None, reverse=False):

        if logdet == 0:
            logdet = torch.zeros_like(z[:, 0, 0, 0])

        # Encode
        if not reverse:
            for layer in self.layers:
                if isinstance(layer, modules.Split):
                    z, logdet, logpz = layer(
                        z, logdet=logdet, logpz=logpz, eps=eps, reverse=False
                    )
                elif isinstance(layer, FlowStep):
                    z, logdet = layer(z, logdet=logdet, reverse=False)
                elif isinstance(layer, modules.Squeeze):
                    z = layer(z, reverse=False)
                elif isinstance(layer, modules.GaussianPrior):
                    z, logdet, logpz = layer(
                        x=z, logdet=logdet, logpz=logpz, eps=eps, reverse=False
                    )

        else:
            # Decode
            for layer in reversed(self.layers):
                if isinstance(layer, modules.Split):
                    z, logdet, logpz = layer(
                        z, logdet=logdet, logpz=logpz, eps=eps, reverse=True
                    )
                elif isinstance(layer, FlowStep):
                    z, logdet = layer(z, logdet=logdet, reverse=True)
                elif isinstance(layer, modules.Squeeze):
                    z = layer(z, reverse=True)
                elif isinstance(layer, modules.GaussianPrior):
                    z, logdet, logpz = layer(
                        x=z, logdet=logdet, logpz=logpz, eps=eps, reverse=True
                    )

        return z, logdet, logpz


class FlowModel(nn.Module):
    def __init__(self, input_shape, filter_size, L, K, bsz, n_bits_x=8):
        super().__init__()
        self.flow = NormFlowNet(
            input_shape=input_shape, filter_size=filter_size, bsz=bsz, K=K, L=L
        )

        self.n_bins = 2 ** n_bits_x

    def forward(
        self, x_image=None, z=None, logdet=0, reverse=False, use_stored=False, eps=None
    ):

        if not reverse:
            return self.normalizing_flow(x_image)

        else:
            return self.inverse_flow(z=z, logdet=logdet, eps=eps)

    def normalizing_flow(self, x_image=None):

        # Dequantize pixels: Discrete -> Continuous
        z, logdet = self._dequantize(x_image, self.n_bins)

        # Push z through flow
        z, logdet, logp_z = self.flow.forward(z=z, logdet=logdet)

        # Loss: Z'ks under Gaussian + sum_logdet
        x_bpd = -(logdet + logp_z) / float(np.log(2) * np.prod(x_image.size()[1:]))

        return z, x_bpd

    def inverse_flow(self, z, eps, logdet=0):
        x = self.flow(z=z, logdet=logdet, eps=eps, reverse=True)
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

    def _sample_images(self, z=None, eps=None):
        # Draw sample from prior
        with torch.no_grad():
            samples = self.inverse_flow(z=z, eps=eps)[0]
        return samples
