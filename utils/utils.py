from datetime import datetime
import numpy as np
import random
import torch
import os

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_model(model_with_config, optimizer, args, modelname=None):
    """
    Function to load PyTorch model state dictionary to resume training.
    Args:
        model_with_config: Instance of the model with configurations (e.g. im_shape_hr, im_shape_lr, device, n_bits_x).
        optimizer: State of the optimizer at saving time.
        model_folder: Name of the folder where the model should be saved.
        model_name: Sequence of parameter configurations under which the model was trained. E.g.: model_name = 'step_0_bsz_1
        data_name: Name of the dataset on which the model was trained on.
    Returns:
        model: PyTorch model loaded with trained weights ready to resume training.
        optimizer: Loaded optimizer.
        step: Step at which to resume training procedure.
    """

    if modelname:
        args.modelname = modelname

    savedir = args.model_path + args.exp_name + "/" + "models" + "/"

    # Load model state dictionary with all information
    state = torch.load(savedir + args.modelname + ".pkl")
    model = model_with_config
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    epoch = state["epoch"]

    print("=> Loaded checkpoint at training Epoch: {}".format(state["epoch"]))

    return model, optimizer, epoch


def save_model(model, epoch, optimizer, args, time=False):
    """
    Function to save trained model in PyTorch to resume training later.
    Args:
        step: Training steps completed so far.
        state_dict: The state dictionary of the trained PyTorch model.
        optimizer: State of the optimizer at saving time.
        model_name: Sequence of parameter configurations under which the model was trained. E.g.: model_name = 'step_0_bsz_1
        data_name: Name of the dataset on which the model was trained on.
        model_path: Name of the folder where the model should be saved.
    """
    # Save state of the model

    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "parallel": args.parallel,
    }

    # Creates folder (or path) if it does not exist.
    savedir = args.model_path + "/" + args.exp_name + "/" + "models/"

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if time:
        modelname = args.modelname + "_{}".format(
            datetime.now().strftime("%Y.%m.%d_%H_%M")
        )
        file_name = savedir + modelname + ".pkl"
    else:
        file_name = savedir + args.modelname + ".pkl"

    # Saves data to a file
    f = open(file_name, "wb")
    torch.save(state, file_name)
    f.close()


def flatten_sum(logps):
    while len(logps.size()) > 1:
        logps = logps.sum(dim=-1)
    return logps


def write_bits_per_dim_list(bits_per_dim_list, path, model_name, data_name):
    """
    Writes the bits per dimension stored in a list during training to a file
    """
    f_name = (
        path + data_name + "/" + "bits_per_dim_" + data_name + "_" + model_name + ".txt"
    )
    with open(f_name, "w") as f:
        f.writelines("%s\n" % value for value in bits_per_dim_list)
    f.close()


def read_bits_per_dim_list(path, model_name):
    """
    Reads a stored bits per dim value list from a file.
    """
    f_name = path + "bits_per_dim_" + model_name[:-1] + ".txt"
    with open(f_name, "r") as f:
        bpd = [curr_value.rstrip() for curr_value in f.readlines()]
    f.close()
    return list(np.array(bpd).astype(np.float))


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print(
        "=> loaded checkpoint of {name} from {path}".format(
            name=model.name, path=(path)
        )
    )

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint["state"])
    epoch = checkpoint["epoch"]
    return epoch


def mean(tensor, dims=None, keepdim=False):
    """
    Functionality to compute mean over desired dimensions.
    Args:
        tensor: PyTorch Tensor.
        dims: Dimensions over which the average should be calculated.
        keepdim: If set to true, output tensor has same dimensions as input tensor.
    Result:
        tensor: Tensor after averaging over desired dimensions.
    """
    if dims is None:
        # Mean over all dimensions
        return torch.mean(tensor)
    else:
        # Mean over selected dimensions only
        if isinstance(dims, int):
            dims = [dims]

        dims = sorted(dims)
        for d in dims:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dims):
                tensor.squeeze_(d - i)
        return tensor


def sum(tensor, dims=None, keepdim=False):
    """
    Functionality to compute sum over desired dimensions.
    Args:
        tensor: PyTorch Tensor.
        dims: Dimensions to be summed over.
        keepdim: If set to true, output tensor has same dimensions as input tensor.
    Result:
        tensor: Tensor after summing over desired dimensions.
    """
    if dims is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dims, int):
            dims = [dims]
        dims = sorted(dims)
        for d in dims:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dims):
                tensor.squeeze_(d - i)
        return tensor
