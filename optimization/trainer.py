from datetime import datetime
import torch
import matplotlib.pyplot as plt

# Utils
import utils
import numpy as np
from  tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

# Evaluation
from evaluate import evaluate, metrics_eval
import test

def trainer(args, train_loader, valid_loader, test_loader, model, optimizer,        device, needs_init=True):

    writer = SummaryWriter('runs/{}'.format(args.exp_name))
    prev_bpd_epoch = np.inf
    logging_step = 0
    step = 0
    bpd_valid = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                step_size=2 * 10**5, gamma=0.5)

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            #for param_group in optimizer.param_groups:
            #    print(param_group['lr'])

            y = item[0]
            x = item[1]

            y, x = y.to(device), x.to(device)

            # plt.figure(figsize=(6, 6))
            # plt.imshow(x[0, :, :, :].permute(1, 2, 0).contiguous().detach().cpu().numpy())
            # plt.margins(5)
            # plt.gcf()
            # plt.savefig('xlr_{}.png'.format(batch_idx))
            # plt.close()
            # #
            # plt.figure(figsize=(6, 6))
            # plt.imshow(y[0, :, :, :].clamp(min=0, max=float(32 - 1)/ float(32)).permute(1, 2, 0).contiguous().detach().cpu().numpy())
            # plt.margins(5)
            # plt.gcf()
            # plt.savefig('xhr_{}.png'.format(batch_idx))
            # plt.close()

            model.train()
            optimizer.zero_grad()

            # # We need to init the underlying module in the dataparallel object
            # For ActNorm layers.
            if needs_init and torch.cuda.device_count() > 1:
                bsz_p_gpu = args.bsz // torch.cuda.device_count()
                _, _ = model.module.forward(
                    x_hr=y[:bsz_p_gpu], xlr=x[:bsz_p_gpu], logdet=0)

            # Forward pass
            z, bpd = model.forward(x_hr=y, xlr=x, logdet=0)
            loss = bpd

            writer.add_scalar('bpd_train', bpd.mean().item(), step)

            # Compute gradients
            loss.mean().backward()

            # Update model parameters using calculated gradients
            optimizer.step()
            scheduler.step()
            step += 1

            print("[{}] Train Step: {:01d}/{}, Bsz = {}, Bits per dim {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), 
                step, args.max_steps,
                args.bsz,
                bpd.mean()))

            if step % args.log_interval == 0:
                with torch.no_grad():
                    if hasattr(model, 'module'):
                        model_without_dataparallel = model.module
                    else:
                        model_without_dataparallel = model

                    #test.test_model(model_without_dataparallel,
                    #               test_loader, args)
                    #writer = metrics_eval(
                    #     model_without_dataparallel,
                    #     test_loader,
                    #     logging_step, writer, args)

                    bpd_valid = evaluate(model_without_dataparallel,
                                        valid_loader, args.exp_name,
                                        '{}'.format(step), args)

                    writer.add_scalar(
                       'bpd_valid', bpd_valid.mean().item(), 
                       logging_step)

                    # Save checkpoint only when bpd lower than previous model
                    if bpd_valid < prev_bpd_epoch:
                        utils.save_model(
                            model_without_dataparallel, epoch, optimizer, args,
                            time=True)
                        prev_bpd_epoch = bpd_valid

                    logging_step += 1

            if step == args.max_steps:
                break

        if step == args.max_steps:
            print("Done Training for {} mini-batch update steps!".format(
                    args.  max_steps))

            if hasattr(model, 'module'):
                    model_without_dataparallel = model.module
            else:
                    model_without_dataparallel = model

            utils.save_model(model_without_dataparallel, epoch, 
                             optimizer, args, time=True)
            print('Saved trained model :)')
            break
