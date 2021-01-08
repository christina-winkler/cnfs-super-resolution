import load_data
import torch
import os
from matplotlib import pyplot as plt
import torchvision
import numpy as np
import metrics


def test_model(model, test_loader, args):

    print("Metric evaluation on {}...".format(args.testset))

    savedir = "runs/{}/snapshots/test_images/{}/".format(args.exp_name, args.testset)

    # storing metrics
    ssim_sample = []
    ssim_0 = []
    ssim_05 = []
    ssim_07 = []
    ssim_08 = []
    ssim_1 = []

    psnr_sample = []
    psnr_0 = []
    psnr_05 = []
    psnr_07 = []
    psnr_08 = []
    psnr_1 = []

    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(test_loader):

            y = item[0]
            x = item[1]
            orig_shape = item[2]
            w, h = orig_shape

            # Push tensors to GPU
            y = y.to("cuda")
            x = x.to("cuda")

            # plt.figure(figsize=(6, 6))
            # plt.imshow(x[0, :, :, :].permute(1, 2, 0).contiguous().detach().cpu().numpy())
            # plt.margins(5)
            # plt.gcf()
            # plt.savefig('x_{}.png'.format(idx))
            # plt.close()

            # plt.figure(figsize=(6, 6))
            # plt.imshow(y[0, :, :, :].permute(1, 2, 0).contiguous().detach().cpu().numpy())
            # plt.margins(5)
            # plt.gcf()
            # plt.savefig('y_{}.png'.format(idx))
            # plt.close()

            if args.modeltype == "flow":

                mu0 = model._sample(x=x, eps=0)
                mu05 = model._sample(x=x, eps=0.5)
                mu07 = model._sample(x=x, eps=0.7)
                mu08 = model._sample(x=x, eps=0.8)
                mu1 = model._sample(x=x, eps=1)

                ssim_0.append(metrics.ssim(y, mu0, orig_shape))
                ssim_05.append(metrics.ssim(y, mu05, orig_shape))
                ssim_07.append(metrics.ssim(y, mu07, orig_shape))
                ssim_08.append(metrics.ssim(y, mu08, orig_shape))
                ssim_1.append(metrics.ssim(y, mu1, orig_shape))

                psnr_0.append(metrics.psnr(y, mu0, orig_shape))
                psnr_05.append(metrics.psnr(y, mu05, orig_shape))
                psnr_07.append(metrics.psnr(y, mu07, orig_shape))
                psnr_08.append(metrics.psnr(y, mu08, orig_shape))
                psnr_1.append(metrics.psnr(y, mu1, orig_shape))

                # ---------------------- Visualize Samples---------------------
                if args.visual:
                    os.makedirs(savedir, exist_ok=True)
                    w, h = orig_shape
                    torchvision.utils.save_image(
                        x,
                        savedir + "{}_x.png".format(idx),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        y[:, :, :h, :w],
                        savedir + "{}_y.png".format(idx),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        mu0[:, :, :h, :w],
                        savedir + "{}_eps{}.png".format(idx, 0),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        mu05[:, :, :h, :w],
                        savedir + "{}_eps{}.png".format(idx, 0.5),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        mu07[:, :, :h, :w],
                        savedir + "{}_eps{}.png".format(idx, 0.7),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        mu08[:, :, :h, :w],
                        savedir + "{}_eps{}.png".format(idx, 0.8),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        mu1[:, :, :h, :w],
                        savedir + "{}_eps{}.png".format(idx, 1),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )

            elif args.modeltype == "dlogistic":

                # sample from model
                sample, means = model._sample(x=x)
                ssim_0.append(metrics.ssim(y, means, orig_shape))
                psnr_0.append(metrics.psnr(y, means, orig_shape))
                ssim_sample.append(metrics.ssim(y, means, orig_shape))
                psnr_sample.append(metrics.psnr(y, means, orig_shape))

                # ---------------------- Visualize Samples-------------
                if args.visual:
                    # only for testing, delete snippet later
                    torchvision.utils.save_image(
                        x[:, :, :h, :w],
                        "{}_x.png".format(idx),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        y[:, :, :h, :w],
                        "{}_y.png".format(idx),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        means[:, :, :h, :w],
                        "{}_mu.png".format(idx),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        sample[:, :, :h, :w],
                        "{}_sample.png".format(idx),
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
        # store metrics
        file = open(
            "runs/{}/metric_eval_{}_{}.txt".format(
                args.exp_name, args.testset, args.modelname
            ),
            "w",
        )

        file.write("ssim mu: {} \n".format(np.mean(ssim_0)))
        # file.write('ssim sample (dlog): {} \n'.format(np.mean(ssim_sample)))
        file.write("ssim mu+05:{} \n".format(np.mean(ssim_05)))
        file.write("ssim mu+07:{} \n".format(np.mean(ssim_07)))
        file.write("ssim mu+08:{} \n".format(np.mean(ssim_08)))
        file.write("ssim mu+1:{} \n".format(np.mean(ssim_1)))

        file.write("psnr mu: {} \n".format(np.mean(psnr_0)))
        # file.write('psnr sample (dlog): {} \n'.format(np.mean(psnr_sample)))
        file.write("psnr mu+05: {} \n".format(np.mean(psnr_05)))
        file.write("psnr mu+07:{} \n".format(np.mean(psnr_07)))
        file.write("psnr mu+08: {} \n".format(np.mean(psnr_08)))
        file.write("psnr mu+1: {} \n".format(np.mean(psnr_1)))
        file.close()

        print(
            "Done testing {} model {} on {} !".format(
                args.modeltype, args.modelname, args.testset
            )
        )
