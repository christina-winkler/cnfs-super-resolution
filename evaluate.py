import numpy as np
import torch
import random

import cv2
import PIL
import os
import torchvision
from torchvision import transforms

from os.path import exists, join
import matplotlib.pyplot as plt
from utils.load_data import PILToTensor, Downsample, is_test_image_file
from utils import load_data, metrics


def evaluate(model, data_loader, exp_name, logstep, args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    bpd_list = []
    lim_count = 0
    lim = int(len(data_loader) * 0.0002)

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(data_loader):

            y = item[0]
            x = item[1]

            # Push tensors to GPU
            y = y.to("cuda")
            x = x.to("cuda")

            if args.modeltype == "dlogistic":
                logp_mass, means, logsigmas = model.forward(y, x)
                ndims = np.prod(y.size()[1:])
                bpd = -logp_mass / (np.log(2) * ndims)

            elif args.modeltype == "flow":
                # Push xhr through Flow:
                z, bpd = model.forward(x_hr=y, xlr=x, logdet=0)

            # Generative loss
            bpd_list.append(bpd.mean().detach().cpu().numpy())

            lim_count += 1
            if lim_count == lim:
                break

        # ---------------------- Sample from model ----------------------
        if args.modeltype == "flow":
            mu0 = model._sample(x=x, eps=0)
            mu05 = model._sample(x=x, eps=0.5)
            mu08 = model._sample(x=x, eps=0.8)
            mu1 = model._sample(x=x, eps=1)
            savedir = "runs/{}/snapshots/sampled_images/{}/".format(
                args.exp_name, args.trainset
            )
            os.makedirs(savedir, exist_ok=True)

            x = x.clamp(min=0, max=float(2 ** args.nbits - 1) / float(2 ** args.nbits))

            y = y.clamp(min=0, max=float(2 ** args.nbits - 1) / float(2 ** args.nbits))

            torchvision.utils.save_image(
                x[:64],
                savedir + "{}_x.png".format(logstep),
                nrow=8,
                padding=2,
                normalize=False,
            )
            torchvision.utils.save_image(
                y[:64],
                savedir + "{}_y.png".format(logstep),
                nrow=8,
                padding=2,
                normalize=False,
            )
            torchvision.utils.save_image(
                mu0[:64],
                savedir + "{}_mu_eps{}.png".format(logstep, 0),
                nrow=8,
                padding=2,
                normalize=False,
            )
            torchvision.utils.save_image(
                mu05[:64],
                savedir + "{}_mu_eps{}.png".format(logstep, 0.5),
                nrow=8,
                padding=2,
                normalize=False,
            )
            torchvision.utils.save_image(
                mu08[:64],
                savedir + "{}_mu_eps{}.png".format(logstep, 0.8),
                nrow=8,
                padding=2,
                normalize=False,
            )
            torchvision.utils.save_image(
                mu1[:64],
                savedir + "{}_mu_eps{}.png".format(logstep, 1),
                nrow=8,
                padding=2,
                normalize=False,
            )

        elif args.modeltype == "dlogistic":
            savedir = "runs/{}/snapshots/sampled_images/{}/".format(
                args.exp_name, args.testset
            )
            os.makedirs(savedir, exist_ok=True)
            torchvision.utils.save_image(
                x[:64],
                savedir + "{}_x.png".format(logstep),
                nrow=8,
                padding=2,
                normalize=False,
            )
            torchvision.utils.save_image(
                y[:64],
                savedir + "{}_y.png".format(logstep),
                nrow=8,
                padding=2,
                normalize=False,
            )
            torchvision.utils.save_image(
                means[:64],
                savedir + "{}_mu.png".format(logstep),
                nrow=8,
                padding=2,
                normalize=False,
            )

    print("Eval bpd mean:", np.mean(bpd_list))
    return np.mean(bpd_list)


def metrics_eval(model, test_loader, logging_step, writer, args):

    print("Metric evaluation on {}...".format(args.testset))

    # storing metrics
    # ssim_yhat = []
    ssim_mu0 = []
    ssim_mu05 = []
    ssim_mu08 = []
    ssim_mu1 = []
    # psnr_yhat = []
    psnr_0 = []
    psnr_05 = []
    psnr_08 = []
    psnr_1 = []

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            y = item[0]
            x = item[1]
            orig_shape = item[2]
            w, h = orig_shape

            # Push tensors to GPU
            y = y.to("cuda")
            x = x.to("cuda")

            if args.modeltype == "flow":
                mu0 = model._sample(x=x, eps=0)
                mu05 = model._sample(x=x, eps=0.5)
                mu08 = model._sample(x=x, eps=0.8)
                mu1 = model._sample(x=x, eps=1)

                ssim_mu0.append(metrics.ssim(y, mu0, orig_shape))
                ssim_mu05.append(metrics.ssim(y, mu05, orig_shape))
                ssim_mu08.append(metrics.ssim(y, mu08, orig_shape))
                ssim_mu1.append(metrics.ssim(y, mu1, orig_shape))

                psnr_0.append(metrics.psnr(y, mu0, orig_shape))
                psnr_05.append(metrics.psnr(y, mu05, orig_shape))
                psnr_08.append(metrics.psnr(y, mu08, orig_shape))
                psnr_1.append(metrics.psnr(y, mu1, orig_shape))

            elif args.modeltype == "dlogistic":
                # sample from model
                sample, means = model._sample(x=x)
                ssim_mu0.append(metrics.ssim(y, means, orig_shape))
                psnr_0.append(metrics.psnr(y, means, orig_shape))

                # ---------------------- Visualize Samples-------------
                if args.visual:
                    # only for testing, delete snippet later
                    torchvision.utils.save_image(
                        x[:, :, :h, :w], "x.png", nrow=1, padding=2, normalize=False
                    )
                    torchvision.utils.save_image(
                        y[:, :, :h, :w], "y.png", nrow=1, padding=2, normalize=False
                    )
                    torchvision.utils.save_image(
                        means[:, :, :h, :w],
                        "dlog_mu.png",
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        sample[:, :, :h, :w],
                        "dlog_sample.png",
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )

        writer.add_scalar("ssim_std0", np.mean(ssim_mu0), logging_step)
        writer.add_scalar("psnr0", np.mean(psnr_0), logging_step)

        if args.modeltype == "flow":
            writer.add_scalar("ssim_std05", np.mean(ssim_mu05), logging_step)
            writer.add_scalar("ssim_std08", np.mean(ssim_mu08), logging_step)
            writer.add_scalar("ssim_std1", np.mean(ssim_mu1), logging_step)
            writer.add_scalar("psnr05", np.mean(psnr_05), logging_step)
            writer.add_scalar("psnr08", np.mean(psnr_08), logging_step)
            writer.add_scalar("psnr1", np.mean(psnr_1), logging_step)

        print("PSNR (GT,mean):", np.mean(psnr_0))
        print("SSIM (GT,mean):", np.mean(ssim_mu0))

        return writer
