from visual_evaluation import plot_samples_test
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import utils
from evaluate import evaluate
import condNF


def resolve_sample(n, model, modelname, data, upsc_fac, grid=False):

    data_loader = DataLoader(data, n, shuffle=True)
    test_im = next(iter(data_loader))[0].to("cuda")
    xlr_test = F.interpolate(
        test_im, scale_factor=1 / upsc_fac, mode="bilinear", align_corners=False
    )
    test_samples = get_samples(model, xlr_test)

    plot_samples_test(
        n, test_samples, test_im, modelname, "model_samples", exp="samples", grid=grid
    )


def plot_model_mean(n, model, modelname, data, s):

    data_loader = DataLoader(data, n, shuffle=True)

    test_im = next(iter(data_loader))[0].to("cuda")
    xlr_test = F.interpolate(
        test_im, scale_factor=1 / s, mode="bilinear", align_corners=False
    )

    test_samples = get_samples(model, xlr_test, eps=0)
    plot_samples_test(
        9,
        test_samples,
        test_im,
        modelname,
        "conditional_distribution_mean",
        exp="dist_mean",
        grid=True,
    )


def get_samples(model, xlr, eps=None):
    model.eval()
    with torch.no_grad():
        samples = model._super_resolve(xlr, eps=eps)
        return samples


def run_on_test_set(model, optimizer, valid, test, modelname):
    f = open("final_valid_set_bpd.txt", "w")
    f.write(
        "Test Set bpd: {:.4f} \n".format(
            evaluate(model, valid, modelname, args.bsz, args.s)
        )
    )
    f.close()
    #
    # Evaluate model on test set
    model, _, _ = utils.load_model(
        model, optimizer, args.model_path, modelname, args.dataset
    )

    f = open("test_set_bpd.txt", "w")
    f.write(
        "Test Set bpd: {:.4f} \n".format(
            evaluate(model, test, modelname, args.bsz, args.s)
        )
    )
    f.close()


def run_experiment(args):

    # Load data
    if args.dataset == "cifar10":
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        )

        full_train_data = torchvision.datasets.CIFAR10(
            root="../data/cifar10", train=True, download=False, transform=transform
        )

        test = torchvision.datasets.CIFAR10(
            root="../data/cifar10", train=False, download=False, transform=transform
        )
        train_size = int(0.8 * len(full_train_data))
        valid_size = len(full_train_data) - train_size
        train, valid = torch.utils.data.random_split(
            full_train_data, [train_size, valid_size]
        )
        im_shape_hr = (3, 32, 32)

    if args.dataset == "imagenet":
        print("Loading ImageNet Data...")
        train32 = torchvision.datasets.ImageFolder(
            root="../imagenet32/train_32x32", transform=transforms.ToTensor()
        )
        valid32 = torchvision.datasets.ImageFolder(
            root="../imagenet32/valid_32x32", transform=transforms.ToTensor()
        )
        train64 = torchvision.datasets.ImageFolder(
            root="../imagenet64/train_64x64", transform=transforms.ToTensor()
        )
        valid64 = torchvision.datasets.ImageFolder(
            root="../imagenet64/valid_64x64", transform=transforms.ToTensor()
        )
        train = {"32": train32, "64": train64}
        valid = {"32": valid32, "64": valid64}
        im_shape_hr = (3, 64, 64)

    # Build name of current model
    modelname = "{}_bsz_{}_K_{}_L{}_lr_{:.4f}".format(
        args.model, args.bsz, args.K, args.L, args.lr
    )

    # Initialize model & Optimizer
    model = condNF.FlowModel(
        im_shape_hr, args.filter_size, args.L, args.K, args.bsz, args.s
    ).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    # Load model to run experiments on
    model, optimizer, epoch = utils.load_model(
        model, optimizer, args.model_path, modelname, "cifar10"
    )

    if args.experiment == "resolve_samples":
        resolve_sample(9, model, modelname, test, args.s, grid=True)
    elif args.experiment == "plot_mean":
        plot_model_mean(9, model, modelname, test, args.s)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="condNF", help="Model you want to train."
    )
    parser.add_argument("--s", type=int, default=2, help="Upscaling factor.")
    parser.add_argument("--crop_sz", type=int, default=32, help="Training patch size.")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--bsz", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--filter_size",
        type=int,
        default=512,
        help="filter size for DeepConvNet in Affine Coupling Layer",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=3,
        help="Number of levels specifying the depth of the model",
    )
    parser.add_argument("--K", type=int, default=8, help="Number of flow steps")
    parser.add_argument(
        "--dataset", type=str, default="cifar10", help="Dataset to train the model on."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/cifar10/cifar-10-batches-py",
        help="Path to directory where data is stored.",
    )
    parser.add_argument(
        "--data_part",
        type=str,
        default="train",
        help="Specify if train, valid or test set should be loaded.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/",
        help="Directory where models are saved.",
    )
    parser.add_argument(
        "--test", type=bool, default=True, help="Model run on test set."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="resolve_samples",
        help="Model run on test set.",
    )

    args = parser.parse_args()

    run_experiment(args)
