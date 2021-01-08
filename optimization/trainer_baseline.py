from datetime import datetime
import numpy as np
import os
import torch

# Utils
import utils

# Training curve, reconstructions and sample plotting
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

# Evaluation
from evaluate import evaluate, metrics_eval


def trainer_baseline(
    args,
    train_loader,
    valid_loader,
    test_loader,
    model,
    optimizer,
    device,
    needs_init=True,
):

    writer = SummaryWriter("runs/{}".format(args.exp_name))
    prev_bpd_epoch = np.inf
    logging_step = 0
    step = 0

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):
            y = item[0]
            x = item[1]

            y, x = y.to(device), x.to(device)

            # y.requires_grad = True
            # x.requires_grad = True

            # plt.figure(figsize=(6, 6))
            # plt.imshow(x[0, :, :, :].permute(1, 2, 0).contiguous().detach().cpu().numpy())
            # plt.margins(5)
            # plt.gcf()
            # plt.savefig('xlr_{}.png'.format(batch_idx))
            # plt.close()
            # #
            # plt.figure(figsize=(6, 6))
            # plt.imshow(y[0, :, :, :].permute(1, 2, 0).contiguous().detach().cpu().numpy())
            # plt.margins(5)
            # plt.gcf()
            # plt.savefig('xhr_{}.png'.format(batch_idx))
            # plt.close()
            # quit()

            model.train()
            optimizer.zero_grad()

            # Forward pass
            logp_mass, _, _ = model.forward(y, x)
            ndims = np.prod(y.size()[1:])
            bpd = -logp_mass / np.log(2) / ndims
            writer.add_scalar("bpd_train", bpd.mean().item(), step)

            # Compute gradients
            bpd.mean().backward()

            # Update model parameters using calculated gradients
            optimizer.step()
            step += 1

            print(
                "[{}] Train Step: {:01d}/{}, Bsz = {}, Average bpd {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    step,
                    args.max_steps,
                    args.bsz,
                    bpd.mean(),
                )
            )

            if step % args.log_interval == 0:
                with torch.no_grad():
                    if hasattr(model, "module"):
                        model_without_dataparallel = model.module
                    else:
                        model_without_dataparallel = model

                    writer = metrics_eval(
                        model_without_dataparallel,
                        test_loader,
                        logging_step,
                        writer,
                        args,
                    )

                    # bpd_valid = evaluate(model_without_dataparallel,
                    #                     valid_loader, args.exp_name,
                    #                     '{}'.format(step), args)

                    writer.add_scalar(
                        "bpd_valid", bpd_valid.mean().item(), logging_step
                    )

                    # Save checkpoint only when bpd lower than previous model
                    if bpd_valid < prev_bpd_epoch:
                        utils.save_model(
                            model_without_dataparallel,
                            epoch,
                            optimizer,
                            args,
                            time=True,
                        )
                        prev_bpd_epoch = bpd_valid

                    logging_step += 1

            if step == args.max_steps:
                break

        if step == args.max_steps:
            print(
                "Done Training for {} mini-batch update steps!".format(args.max_steps)
            )

            if hasattr(model, "module"):
                model_without_dataparallel = model.module
            else:
                model_without_dataparallel = model
            utils.save_model(
                model_without_dataparallel, epoch, optimizer, args, time=True
            )
            print("Saved trained model :)")
            break
