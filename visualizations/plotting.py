import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def bits_per_dim_training_curve(bpd_train, bpd_valid, modelname):
    """
    Plots average bits per pixel channel for a sample image from the training and validation set
    over training steps.
    x-axis: Training steps.
    y-axis: Bits per dimension
    """
    savedir = 'snapshots/{}/train_curve/'.format(modelname)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    x_ticks = np.arange(len(bpd_train))
    ax.plot(x_ticks, bpd_train, color='black', label='train')
    ax.plot(x_ticks, bpd_valid, color='blue', label='valid')
    ax.grid()

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Bits per dim')

    fig.legend(loc='upper right')
    fig.savefig(savedir + 'train_curve')
    plt.close()
