import os
import torch
import torchvision
import random
import numpy as np
import cv2

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib.pyplot as plt


def vis_eval_test_images(y_hat, title, args, eps=0):
    os.makedirs(
        "runs/{}/snapshots/test_images/{}/".format(args.exp_name, args.testset),
        exist_ok=True,
    )

    # todo: check how to get image file name form dataloader
    savedir = "runs/{}/snapshots/test_images/{}/{}_eps{}.png".format(
        args.exp_name, args.testset, title, eps
    )

    with torch.no_grad():
        y_hat = y_hat.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        y_hat = np.transpose(y_hat[[2, 1, 0], :, :], (1, 2, 0))
        y_hat = (y_hat * 256.0).round()
        cv2.imwrite(savedir, y_hat)


def vis_eval_plotting(x, y, exp_name, logging_step, plot_type, eps=0):

    if not os.path.exists("runs/{}/snapshots/".format(exp_name)):
        os.makedirs("runs/{}/snapshots/".format(exp_name))

    left_title = "x"
    if plot_type == "reconstruction":
        savedir = "runs/{}/snapshots/step{}_reconstruction.png".format(
            exp_name, logging_step
        )
        right_title = "y_tilde"
        left_title = "y"
    elif plot_type == "samples":
        savedir = "runs/{}/snapshots/step{}_samples.png".format(exp_name, logging_step)
        right_title = "y_hat"
        left_title = "x"
    elif plot_type == "mu+eps":
        savedir = "runs/{}/snapshots/step{}_mu+{}.png".format(
            exp_name, logging_step, eps
        )
        right_title = "mu|x+{}".format(eps)
        left_title = "x"

    with torch.no_grad():

        grid1 = torchvision.utils.make_grid((x[0:9, :, :, :]).cpu(), nrow=3)
        grid2 = torchvision.utils.make_grid(y[0:9, :, :, :].cpu(), nrow=3)

        plt.figure()
        # plt.axis('off')
        plt.subplot(1, 2, 1)
        plt.title(left_title)
        plt.imshow(grid1.permute(1, 2, 0).contiguous())
        # plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title(right_title)
        plt.imshow(grid2.permute(1, 2, 0).contiguous())
        # plt.axis('off')
        plt.savefig(savedir)
        plt.close()


def plot_samples_test(
    n, z_sample, x_lr, modelname, datapart, plot_type=None, grid=False
):
    if not os.path.exists("snapshots/super_resolved/"):
        os.makedirs("snapshots/super_resolved/")

    savedir = "snapshots/super_resolved/{}_super_resolved_{}.png".format(
        datapart, modelname
    )

    left_head = "Ground truth"

    if plot_type == "samples":
        right_head = "Super-resolved"
    elif plot_type == "dist_mean":
        right_head = "CondDistr Mean - MU|xlr"

    if not grid:
        plt.figure()
        for i in range(n):
            plt.subplot(1, 2, 1)
            plt.title(left_head)
            grid1 = torchvision.utils.make_grid(x_lr.cpu(), nrow=1)
            plt.imshow(grid1.permute(1, 2, 0).contiguous())
            plt.subplot(1, 2, 2)
            plt.title(right_head)
            grid2 = torchvision.utils.make_grid(z_sample[i, :, :, :].cpu(), nrow=1)
            # plt.axis('off')
            plt.imshow(grid2.permute(1, 2, 0).contiguous())
            savedir = "snapshots/super_resolved/{}_{}_super_resolved_{}.png".format(
                i, datapart, modelname
            )
            # plt.axis('off')
            plt.savefig(savedir)

    else:
        with torch.no_grad():
            grid1 = torchvision.utils.make_grid(x_lr.cpu(), nrow=3, normalize=True)
            grid2 = torchvision.utils.make_grid(z_sample[0:n, :, :, :].cpu(), nrow=3)
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.title(left_head)
            # plt.axis('off')
            plt.imshow(grid1.permute(1, 2, 0).contiguous())
            plt.subplot(1, 2, 2)
            plt.title(right_head)
            # plt.axis('off')
            plt.imshow(grid2.permute(1, 2, 0).contiguous())
            # plt.axis('off')
            plt.savefig(savedir)
    plt.close()
