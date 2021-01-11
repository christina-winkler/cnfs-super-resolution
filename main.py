import argparse
import torch
import torch.optim as optim

# Dataset loading
from utils import load_data

# Utils
import utils
import random
import numpy as np

# Models
from models.architectures import condNF
from models.architectures import dlogistic_nn

# Optimization
from optimization import trainer
from optimization import trainer_baseline
import evaluate
import test

from tensorboardX import SummaryWriter
###############################################################################


def main(args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize device on which to run the model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        args.num_gpus = torch.cuda.device_count()
        args.parallel = False
    else:
        device = "cpu"

    # Build name of current model
    if args.modelname is None:
        args.modelname = "{}_{}_bsz{}_K{}_L{}_lr{:.4f}_s{}".format(
            args.model, args.trainset, args.bsz, args.K, args.L, args.lr, args.s
        )

    if args.train:
        # Load data
        if "imagenet" in args.trainset:
            assert "imagenet" in args.testset
            train_loader, valid_loader, test_loader, args = load_data.load_train(args)
        else:
            train_loader, valid_loader, args = load_data.load_train(args)
            test_loader, args = load_data.load_test(args)

    quit()

    # Create model
    print("Creating model..")
    if args.modeltype == "flow":
        model = condNF.FlowModel(
            (3, args.crop_size, args.crop_size),
            args.filter_size,
            args.L,
            args.K,
            args.bsz,
            args.s,
            args.nb,
            args.condch,
            args.nbits,
            args.noscale,
            args.noscaletest,
        )

    # params_flow = sum(x.numel() for x in model.parameters() if x.requires_grad)
    # print('Flow:   ', params_flow)

    if args.modeltype == "dlogistic_nn":
        model = dlogistic_nn.DLogistic_NN(
            args.condch,
            (3, args.crop_size, args.crop_size),
            args.s,
            args.L,
            args.K,
            args.bsz,
            args.nb,
            args.nbits,
        )

    # params_baseline = sum(x.numel() for x in model.parameters() if x.requires_grad)
    # print('Baseline', params_baseline)

    if torch.cuda.device_count() > 1 and args.train:
        print("Running on {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        args.parallel = True

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    if args.train:
        if args.resume_training:

            # Load model if training should be resumed
            model, optimizer, epoch = utils.load_model(
                model, optimizer, args.model_path, args.modelname, args.trainset
            )

        # Start training
        print("Start training {} on {}:".format(args.modeltype, args.trainset))
        if args.modeltype == "dlogistic":
            trainer_baseline.trainer_baseline(
                args, train_loader, valid_loader, test_loader, model, optimizer, device
            )

        if args.modeltype == "flow":
            trainer.trainer(
                args, train_loader, valid_loader, test_loader, model, optimizer, device
            )

    if args.test:

        # Load model
        print("Loading model {}..".format(args.modeltype))
        if args.modeltype == "flow":
            model, optimizer, epoch = utils.load_model(
                model, optimizer, args, args.modelname
            )
            if hasattr(model, "module"):
                print("removing module")
                model_without_dataparallel = model.module
            else:
                model_without_dataparallel = model

        if args.modeltype == "dlogistic":
            print("Loading trained dlogistic ...")
            model = dlogistic_nn.DLogistic_NN(
                args.condch,
                (3, args.crop_size, args.crop_size),
                args.s,
                args.L,
                args.K,
                args.bsz,
                args.nb,
                args.nbits,
            )

            model, optimizer, epoch = utils.load_model(
                model, optimizer, args, args.modelname
            )
            if hasattr(model, "module"):
                print("removing module")
                model_without_dataparallel = model.module
            else:
                model_without_dataparallel = model

        model.to(device)

        # Load data
        if "imagenet" in args.trainset:
            train_loader, valid_loader, test_loader, args = load_data.load_train(args)

            print("BPD Evaluating on ImageNet test set:")
            evaluate.evaluate(model, test_loader, args.exp_name, 0, args)

            test_loader, args = load_data.load_test(args)
            print("BPD Evaluating on testset: {}".format(args.testset))
            evaluate.evaluate(model, test_loader, args.exp_name, 0, args)

        else:
            train_loader, valid_loader, args = load_data.load_train(args)

            test_loader, args = load_data.load_test(args)

            # print("Evaluating on DIV2K validation set:")
            # evaluate.evaluate(model, valid_loader, args.exp_name, 0, args)

            # print("Evaluating on testset: {}".format(args.testset))
            # evaluate.evaluate(model, test_loader, args.exp_name, 0, args)

        print("PSNR/SSIM evaluation ...")
        test.test_model(model, test_loader, args)
        # print("Evaluating on DIV2K validation set:")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # train configs
    parser.add_argument("--model", type=str, default="condNF",
                        help="Model you want to train.")
    parser.add_argument("--modeltype", type=str, default="flow",
                        help="Specify modeltype you would like to train.")
    parser.add_argument("--model_path", type=str, default="runs/",
                        help="Directory where models are saved.")
    parser.add_argument("--modelname", type=str, default=None,
                        help="Sepcify modelname to be tested.")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="number of epochs")
    parser.add_argument("--max_steps", type=int, default=20000,
                        help="For training on a large dataset.")
    parser.add_argument("--log_interval", type=int, default=800,
                        help="Interval in which results should be logged.")

    # runtime configs
    parser.add_argument("--visual", action="store_true",
                        help="Visualizing the samples at test time.")
    parser.add_argument("--noscaletest", action="store_true",
                        help="Disable scale in coupling layers only at test time.")
    parser.add_argument("--noscale", action="store_true",
                        help="Disable scale in coupling layers.")
    parser.add_argument("--test", action="store_true",
                        help="Model run on test set.")
    parser.add_argument("--train", action="store_true",
                        help="If model should be trained.")
    parser.add_argument("--resume_training", action="store_true",
                        help="If training should be resumed.")

    # hyperparameters
    parser.add_argument("--nbits", type=int, default=8,
                        help="Images converted to n-bit representations.")
    parser.add_argument("--s", type=int, default=2, help="Upscaling factor.")
    parser.add_argument("--crop_size", type=int, default=500,
                        help="Crop size when random cropping is applied.")
    parser.add_argument("--patch_size", type=int, default=500,
                        help="Training patch size.")
    parser.add_argument("--bsz", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate")
    parser.add_argument("--filter_size", type=int, default=512,
                        help="filter size NN in Affine Coupling Layer")
    parser.add_argument("--L", type=int, default=2, help="# of levels")
    parser.add_argument("--K", type=int, default=8,
                        help="# of flow steps, i.e. model depth")
    parser.add_argument("--nb", type=int, default=16,
                        help="# of residual-in-residual blocks LR network.")
    parser.add_argument("--condch", type=int, default=128,
                        help="# of residual-in-residual blocks in LR network.")

    # data
    parser.add_argument("--datadir", type=str, default="../data",
                        help="Dataset to train the model on.")
    parser.add_argument("--trainset", type=str, default="cifar10",
                        help="Dataset to train the model on.")
    parser.add_argument("--testset", type=str, default="set5",
                        help="Specify test dataset")
    # experiments
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Name of the experiment.")

    args = parser.parse_args()
    main(args)
