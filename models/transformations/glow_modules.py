import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import utils
import random


class ActNorm(nn.Module):
    """
    Activation Normalization layer which normalizes the activation values of a batch
    by their mean and variance. The activations of each channel then should have
    zero mean and unit variance. This layer ensure more stable parameter updates during
    training as it reduces the variance over the samples in a mini batch.
    from: https://github.com/pclucas14/pytorch-glow/blob/master/invertible_layers.py
    """

    def __init__(self, num_features, logscale_factor=1.0, scale=1.0):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter("b", nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(1, num_features, 1)))

    def forward(self, input, logdet, reverse=False):

        if not reverse:

            input_shape = input.size()
            input = input.view(input_shape[0], input_shape[1], -1)

            if not self.initialized:
                self.initialized = True
                unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()

                # Compute the mean and variance
                sum_size = input.size(0) * input.size(-1)
                b = -torch.sum(input, dim=(0, -1)) / sum_size
                vars = unsqueeze(
                    torch.sum((input + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size
                )
                logs = (
                    torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
                    / self.logscale_factor
                )

                self.b.data.copy_(unsqueeze(b).data)
                self.logs.data.copy_(logs.data)
            logs = self.logs * self.logscale_factor
            b = self.b

            output = (input + b) * torch.exp(logs)
            dlogdet = torch.sum(logs) * input.size(-1)  # c x h

            return output.view(input_shape), logdet + dlogdet

        elif reverse == True:
            assert self.initialized
            input_shape = input.size()
            input = input.view(input_shape[0], input_shape[1], -1)
            logs = self.logs * self.logscale_factor
            b = self.b
            output = input * torch.exp(-logs) - b
            dlogdet = torch.sum(logs) * input.size(-1)  # c x h

            return output.view(input_shape), logdet - dlogdet


class Invert1x1Conv(nn.Conv2d):
    # from https://github.com/pclucas14/pytorch-glow/blob/master/invertible_layers.py
    def __init__(self, num_channels):
        self.num_channels = num_channels
        nn.Conv2d.__init__(self, num_channels, num_channels, 1, bias=False)

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn(self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype("float32"))
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward_(self, x, logdet, reverse=False):

        if not reverse:
            dlogdet = (
                torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1)
            )
            logdet += dlogdet
            output = F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            dlogdet = (
                torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1)
            )
            logdet -= dlogdet
            weight_inv = (
                torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
            )
            output = F.conv2d(
                x,
                weight_inv,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return output, logdet


class Squeeze(nn.Module):
    # taken from: https://github.com/pclucas14/pytorch-glow
    def __init__(self, factor=2):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(
            factor, int
        ), "no point of using this if factor <= 1"
        self.factor = factor

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(
            bs, c * self.factor * self.factor, h // self.factor, w // self.factor
        )

        return x

    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x

    def forward(self, x, reverse=False):
        if len(x.size()) != 4:
            raise NotImplementedError
        if not reverse:
            return self.squeeze_bchw(x)
        else:
            return self.unsqueeze_bchw(x)


class AffineCoupling(nn.Module):
    def __init__(self, in_channels, filter_size=512):
        super().__init__()

        self.Net = nn.Sequential(
            conv2d_actnorm(in_channels // 2, filter_size, filter_size=3, padding=1),
            nn.ReLU(),
            conv2d_actnorm(filter_size, filter_size, filter_size=1, padding=0),
            nn.ReLU(),
            conv2d_zeros(filter_size, in_channels, filter_size=3, padding=1),
        )

        self.Net[0].bias.data.zero_()
        self.Net[2].weight.data.normal_(0, 0.05)
        self.Net[2].bias.data.zero_()

    def forward(self, input, logdet=0, reverse=False):
        if not reverse:
            x1, x2 = utils.split(input)
            h = self.Net(x1)
            t, h_scale = utils.cross(h)
            scale = torch.nn.functional.softplus(h_scale)
            logscale = torch.log(scale)
            y2 = (x2 + t) * scale
            y1 = x1
            y = torch.cat((y1, y2), 1)
            logdet += utils.flatten_sum(logscale)
            return y, logdet
        else:
            y1, y2 = utils.split(input)
            h = self.Net(y1)
            t, h_scale = utils.cross(h)
            scale = torch.nn.functional.softplus(h_scale)
            logscale = torch.log(scale)
            x2 = (y2 / scale) - t
            x1 = y1
            x = torch.cat((x1, x2), 1)
            logdet += utils.flatten_sum(logscale)
            return x, logdet


class conv2d_actnorm(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=None):
        super().__init__(
            channels_in, channels_out, filter_size, stride=stride, padding=padding
        )
        padding = (filter_size - 1) // 2 or padding
        self.conv = nn.Conv2d(
            channels_in,
            channels_out,
            filter_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.actnorm = ActNorm(channels_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.actnorm.forward(x, -1)[0]
        return x


class conv2d_zeros(nn.Conv2d):
    def __init__(
        self,
        channels_in,
        channels_out,
        filter_size=3,
        stride=1,
        padding=0,
        logscale=3.0,
    ):
        super().__init__(
            channels_in, channels_out, filter_size, stride=stride, padding=padding
        )
        self.register_parameter("logs", nn.Parameter(torch.zeros(channels_out, 1, 1)))
        self.logscale_factor = logscale

    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        out = super().forward(input)
        return out * torch.exp(self.logs * self.logscale_factor)


class Split(nn.Module):
    # from https://github.com/pclucas14/pytorch-glow/blob/master/invertible_layers.py
    def __init__(self, C):
        super(Split, self).__init__()
        self.conv_zero = conv2d_zeros(C // 2, C, padding=1)

    def split2d_prior(self, x):
        h = self.conv_zero(x)
        mean, logsd = h[:, 0::2], h[:, 1::2]
        return Gaussian_Diag(mean, logsd)

    def forward(self, x, logdet, logpz, eps, reverse=False, use_stored=False):

        if not reverse:
            z1, y = torch.chunk(x, 2, 1)
            self.y = y.detach()  # keep track of latent variables
            pz = self.split2d_prior(z1)
            logpz += pz.logp(y)
            return z1, logdet, logpz

        else:
            pz = self.split2d_prior(x)
            z2 = self.y if use_stored else pz.sample(eps)
            z = torch.cat((x, z2), 1)
            logpz -= pz.logp(z2)
            return z, logdet, logpz


class GaussianPrior(nn.Module):
    # https://github.com/pclucas14/pytorch-glow/blob/master/invertible_layers.py
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x, logdet, logpz, eps, reverse=False):
        if not reverse:
            mean_and_logsd = torch.cat([torch.zeros_like(x) for _ in range(2)], dim=1)
            mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
            pz = Gaussian_Diag(mean, logsd)
            logpz += pz.logp(x)
            return x, logdet, logpz
        else:
            bsz, c, h, w = self.input_shape

            mean_and_logsd = torch.cuda.FloatTensor(bsz, 2 * c, h, w).fill_(0.0)
            mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
            pz = Gaussian_Diag(mean, logsd)
            z = pz.sample(eps) if x is None else x
            logpz -= pz.logp(z)
            return z, logdet, logpz


# Distributions


def Standard_Gaussian(shape):
    mean, logsd = [torch.cuda.FloatTensor(shape).fill_(0.0) for _ in range(2)]
    return Gaussian_Diag(mean, logsd)


def Gaussian_Diag(mean, logsd):
    class o(object):
        log2pi = float(np.log(2 * np.pi))
        pass

        @staticmethod
        def log_like_i(x):
            return -0.5 * (
                o.log2pi + 2.0 * logsd + ((x - mean) ** 2) / torch.exp(2.0 * logsd)
            )

        @staticmethod
        def sample(eps):
            if eps is None:
                eps = torch.zeros_like(mean).normal_()
            return mean + torch.exp(logsd) * eps

    o.logp = lambda x: utils.sum(o.log_like_i(x), dims=[1, 2, 3])
    return o
