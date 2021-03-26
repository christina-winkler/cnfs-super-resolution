import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import random

from ..distributions.distributions import Gaussian_Diag

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ActNorm(nn.Module):
    """
    Activation Normalization layer which normalizes the activation
    values of a batch by their mean and variance. The activations of
    each channel then should have zero mean and unit variance. This
    layer ensure more stable parameter updates during training as it reduces
    the variance over the samples in a mini batch.
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
            # assert self.initialized
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
        self.init = False

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn(self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype("float32"))
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward_(self, x, logdet, reverse=False):
        _, _, height, width = x.size()

        if not reverse:
            dlogdet = torch.slogdet(self.weight.squeeze())[1] * height * width
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
            dlogdet = torch.slogdet(self.weight.squeeze())[1] * height * width
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

        self.init = True
        return output, logdet


class Shuffle(nn.Module):
    # Shuffling on the channel axis
    def __init__(self, num_channels):
        super(Shuffle, self).__init__()
        indices = np.arange(num_channels)
        np.random.shuffle(indices)
        rev_indices = np.zeros_like(indices)
        for i in range(num_channels):
            rev_indices[indices[i]] = i

        indices = torch.from_numpy(indices).long()
        rev_indices = torch.from_numpy(rev_indices).long()
        self.register_buffer("indices", indices)
        self.register_buffer("rev_indices", rev_indices)
        # self.indices, self.rev_indices = indices.cuda(), rev_indices.cuda()

    def forward(self, x, logdet, reverse=False):
        if not reverse:
            return x[:, self.indices], logdet
        else:
            return x[:, self.rev_indices], logdet


class Squeeze(nn.Module):
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


class Net(nn.Module):
    def __init__(
        self,
        level,
        s,
        in_channels,
        input_shape,
        cond_channels,
        noscale,
        noscaletest,
        intermediate_size=512,
    ):
        super().__init__()

        self.squeezer = Squeeze()
        self.s = s
        self.level = level
        c, w, h = input_shape
        self.cond_channels = cond_channels
        d = 2 if noscale else 1

        self.Net = nn.Sequential(
            nn.Conv2d(
                in_channels // 2 + self.cond_channels, intermediate_size, 3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(intermediate_size, intermediate_size, kernel_size=1),
            nn.ReLU(),
            conv2d_zeros(intermediate_size, in_channels // d, padding=1),
        )

        self.Net[0].bias.data.zero_()
        self.Net[2].weight.data.normal_(0, 0.05)
        self.Net[2].bias.data.zero_()

    def forward(self, input, lr_feat_map):
        h = torch.cat((input, lr_feat_map), 1)
        out = self.Net(h)
        return out


class ConditionalCoupling(nn.Module):
    def __init__(
        self,
        level,
        s,
        in_channels,
        input_shape,
        cond_channels,
        filter_size,
        noscale,
        noscaletest,
    ):
        super().__init__()
        self.Net = Net(
            level, s, in_channels, input_shape, cond_channels, noscale, filter_size
        )
        self.noscale = noscale
        self.noscaletest = noscaletest

    def forward(self, z, lr_feat_map=None, logdet=0, logpz=0, reverse=False):

        z1, z2 = split(z)
        h = self.Net(z1, lr_feat_map)

        if self.noscale:
            # print("Scale disabled")
            t, scale = h, torch.ones_like(h)
        else:
            # print("Scale enabled")
            t, h_scale = cross(h)
            scale = torch.nn.functional.softplus(h_scale)
            logscale = torch.log(scale)

            if self.noscaletest:
                # print("Scale disabled for sampling")
                scale = torch.ones_like(scale)

        # add if testnocsale then t, scale = h, torch.ones_like(h)
        if not reverse:
            y2 = (z2 * scale) + t
            y1 = z1
            logdet += 0 if self.noscale else flatten_sum(logscale)

        else:
            y2 = (z2 - t) / scale
            y1 = z1
            logdet -= 0 if self.noscale else flatten_sum(logscale)

        y = torch.cat((y1, y2), 1)
        return y, logdet


class GaussianPrior(nn.Module):
    def __init__(self, C, s, cond_channels, flow_var_shape, final=False):
        super(GaussianPrior, self).__init__()
        self.flow_var_shape = flow_var_shape
        self.cond_channels = cond_channels
        self.final = final
        self.squeezer = Squeeze()
        self.prior = Gaussian_Diag()
        self.s = s

        if final:
            self.conv = conv2d_zeros(self.cond_channels, 2 * C, padding=1)
        else:
            self.conv = conv2d_zeros(self.cond_channels + C // 2, C, padding=1)

    def split2d_prior(self, z, lr_feat_map):
        x = torch.cat((z, lr_feat_map), 1)
        h = self.conv(x)
        mean, sigma = h[:, 0::2], nn.functional.softplus(h[:, 1::2])
        return mean, sigma

    def final_prior(self, lr_feat_map):
        h = self.conv(lr_feat_map)
        mean, sigma = h[:, 0::2], nn.functional.softplus(h[:, 1::2])
        return mean, sigma

    def forward(
        self, x, lr_feat_map, eps, reverse, logpz=0, logdet=0, use_stored=False
    ):

        if not reverse:
            if not self.final:
                z, y = torch.chunk(x, 2, 1)
                mean, sigma = self.split2d_prior(z, lr_feat_map)
                logpz += self.prior.logp(y, mean, sigma)
            else:
                # final prior computation
                mean, sigma = self.final_prior(lr_feat_map)
                logpz += self.prior.logp(x, mean, sigma)
                z = x
        else:
            if not self.final:
                mean, sigma = self.split2d_prior(x, lr_feat_map)
                z2 = self.prior.sample(mean, sigma, eps=eps)
                z = torch.cat((x, z2), 1)

            else:
                # final prior computation
                self.bsz = lr_feat_map.size()[0]
                _, c, h, w = self.flow_var_shape
                mean, sigma = self.final_prior(lr_feat_map)
                z = self.prior.sample(mean, sigma, eps=eps)

        return z, logdet, logpz

########################### UTILS ##############################################


def split(feature):
    """
    Splits the input feature tensor into two halves along the channel dimension.
    Channel-wise masking.
    Args:
        feature: Input tensor to be split.
    Returns:
        Two output tensors resulting from splitting the input tensor into half
        along the channel dimension.
    """
    C = feature.size(1)
    return feature[:, : C // 2, ...], feature[:, C // 2:, ...]


def cross(feature):
    """
    Performs two different slicing operations along the channel dimension.
    Args:
        feature: PyTorch Tensor.
    Returns:
        feature[:, 0::2, ...]: Selects every feature with even channel dimensions index.
        feature[:, 1::2, ...]: Selects every feature with uneven channel dimension index.
    """
    return feature[:, 0::2, ...], feature[:, 1::2, ...]


def concat_feature(tensor_a, tensor_b):
    """
    Concatenates features along the first dimension.
    """
    return torch.cat((tensor_a, tensor_b), dim=1)


def flatten_sum(logps):
    while len(logps.size()) > 1:
        logps = logps.sum(dim=-1)
    return logps
