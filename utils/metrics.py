from skimage.measure import compare_ssim, compare_psnr, compare_nrmse
from utils.metrics_esrgan import calculate_ssim, calculate_psnr, bgr2ycbcr
import torchvision
import numpy as np


def ssim(im1, im2, orig_shape):
    """
    Computes the similarity index between two images measuring
    the similarity between the two images. SSIM has a maximum value of 1, indicating that the two signals are perfectly structural similar
    while a value of 0 indicates no structural similarity.

    Args:
        im1 (tensor):
        im2 (tensor):
    Returns:
        ssim (tensor):
    """
    # crop images back to original shape
    w, h = orig_shape
    im1_orig = im1[:, :, :h, :w]
    im2_orig = im2[:, :, :h, :w]

    # torchvision.utils.save_image(
    #                     im1_orig, 'runs/y{}.png'.format(idx),nrow=1, padding=2, normalize=False)
    # torchvision.utils.save_image(
    #                     im2_orig, 'runs/yhat{}.png'.format(idx),
    #                     nrow=1, padding=2, normalize=False)

    im1 = im1_orig.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    im2 = im2_orig.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()

    ssim = []
    
    # Compute ssim over samples in mini-batch
    for i in range(im1.shape[0]):
        ssim.append(calculate_ssim(im1[i, :, :, :] * 255, im2[i, :, :, :] * 255))
    return np.mean(ssim)


def psnr(im1, im2, orig_shape):
    """

    Args:

    Returns:

    """
    # crop images back to original shape
    w, h = orig_shape
    im1_orig = im1[:, :, :h, :w]
    im2_orig = im2[:, :, :h, :w]
    im1 = im1_orig.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    im2 = im2_orig.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    psnr = []
    # Compute ssim over samples in mini-batch
    for i in range(im1.shape[0]):
        psnr.append(calculate_psnr(im1[i, :, :, :] * 255, im2[i, :, :, :] * 255))
    return np.mean(psnr)


def nrmse(im1, im2):
    """

    Args:

    Returns:

    """
    im1 = im1.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    im2 = im2.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    nrmse = []
    # Compute ssim over samples in mini-batch
    for i in range(im1.shape[0]):
        nrmse.append(compare_nrmse(im1[i, :, :, :], im2[i, :, :, :]))
    return np.mean(nrmse)
