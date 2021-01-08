import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch


class Gaussian_Diag(object):
    def __init__(self):
        super().__init__()
        pass

    def logp(self, x, mean, sigma):
        ones = torch.ones_like(x)
        ll = -0.5 * (x - mean) ** 2 / (sigma ** 2) - 0.5 * torch.log(
            2 * np.pi * (sigma ** 2) * ones
        )
        return torch.sum(ll, [1, 2, 3])

    def sample(self, mean, sigma, eps=0):
        noise = torch.randn_like(mean)
        return mean + eps * sigma * noise


class DiscLogistic(object):
    """
    Class Discretized Logistic Distribution.
    """

    def __init__(self):
        super(DiscLogistic).__init__()
        pass

    def log_pmass(self, y, means, logsigmas, num_classes, log_scale_min=-7.0):
        # from: https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
        # and https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py

        # compute log probability mass
        centered_y = y - means
        inv_std = torch.exp(-logsigmas)
        plus_in = inv_std * (centered_y + 1.0 / num_classes)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_std * (centered_y - 1.0 / num_classes)
        cdf_min = torch.sigmoid(min_in)

        max_condition = (y > 0.9999 - 1 / 256.0).float()
        min_condition = (y < 0.0001).float()

        cdf_plus = 1 * max_condition + cdf_plus * (1 - max_condition)
        cdf_min = 0 * min_condition + cdf_min * (1 - min_condition)

        log_pmass = torch.log(cdf_plus - cdf_min + 1e-8)
        return torch.sum(log_pmass, [1, 2, 3])

    def sample_logistic(
        self, means, logscales, log_scale_min=-7.0, clamp_log_scale=False
    ):

        if clamp_log_scale:
            logscales = torch.clamp(logscales, min=log_scale_min)

        # (1e-5, 1 - 1e-5)  to avoid saturation regions of the sigmoid
        # u = means.data.new(means.size()).uniform_(1e-5, 1. - 1e-5)
        u = means.data.new(means.size()).uniform_(1.0, 0)
        scale = torch.exp(logscales)
        sample = means + scale * torch.log(u / (1.0 - u))
        sample = sample.clamp(min=0, max=float(256.0 - 1.0) / float(256.0))
        return sample, means
