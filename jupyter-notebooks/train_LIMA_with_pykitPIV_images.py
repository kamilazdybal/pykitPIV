# Standard
import argparse
import os

import pytorch_lightning as pl
import torchvision.transforms
from rich import print
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import h5py
import skimage.io as io
import torch
import math
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import cmcrameri.cm as cmc

# pykitPIV
from pykitPIV import Particle, FlowField, Motion, Image

# LIMA
import lima
import lima.dataset

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Generate pykitPIV image pairs

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

image_size = (128,128)
size_buffer = 10
figsize = (5,3)

def generate_images(n_images, random_seed):

    # Instantiate an object of the Particle class:
    particles = Particle(n_images,
                         size=image_size,
                         size_buffer=size_buffer,
                         diameters=(4,4.1),
                         distances=(1,2),
                         densities=(0.05,0.1),
                         signal_to_noise=(5,20),
                         diameter_std=0.2,
                         seeding_mode='random',
                         random_seed=random_seed)

    # Instantiate an object of the FlowField class:
    flowfield = FlowField(n_images,
                          size=image_size,
                          size_buffer=size_buffer,
                          flow_mode='random',
                          gaussian_filters=(10,11),
                          n_gaussian_filter_iter=20,
                          sin_period=(30,300),
                          displacement=(0,10),
                          random_seed=random_seed)

    # Instantiate an object of the Motion class:
    motion = Motion(particles, 
                    flowfield, 
                    time_separation=0.1)

    # Instantiate an object of the Image class:
    image = Image(random_seed=random_seed)

    # Prepare images - - - - - - - - - - - - - - - - - - 

    image.add_particles(particles)

    image.add_velocity_field(flowfield)
            
    motion.forward_euler(n_steps=10)
    
    image.add_motion(motion)
    
    image.add_reflected_light(exposures=(0.6,0.65),
                              maximum_intensity=2**16-1,
                              laser_beam_thickness=1,
                              laser_over_exposure=1,
                              laser_beam_shape=0.95,
                              alpha=1/10)

    image.remove_buffers()

    return image

# Training samples - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

n_images = 100
training_random_seed = 100

image_train = generate_images(n_images, training_random_seed)

image_pairs_train = image_train.image_pairs_to_tensor()
targets_train = image_train.targets_to_tensor()

# Testing samples - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

n_images = 10
test_random_seed = 200

image_test = generate_images(n_images, test_random_seed)

image_pairs_test = image_test.image_pairs_to_tensor()
targets_test = image_test.targets_to_tensor()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def argument_parser():
    """Define parameters that only apply to this model/training"""
    # Hyper-parameters and settings
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training settings
    parser.add_argument(
        "--batch_size", type=int, default=4, metavar="N", help="training batch size"
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="base starting learning rate",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.5,
        metavar="GAMMA",
        help="learning rate#  scheduler gamma",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=4e-4,
        metavar="W",
        help="weight decay parameter",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="BETA1", help="SGD momentum"
    )
    parser.add_argument(
        "--beta",
        default=0.999,
        type=float,
        metavar="BETA2",
        help="beta parameter for adam",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="*",
        metavar=("N1", "N2"),
        default=[60, 80, 100, 120, 140],
        help="epochs at which learning rate scaled by learning rate `gamma`",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of warmup epochs",
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", metavar="OPT", help="optimizer"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ReduceLROnPlateau",
        metavar="SCH",
        help="scheduler",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, metavar="N", help="number of epochs to train"
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.2,
        metavar="GAMMA",
        help="Learning rate decay for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        metavar="N",
        help="Patience level for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0,
        metavar="STD",
        help="Noise std for random noise",
    )
    parser.add_argument(
        "--output_level",
        type=int,
        default=6,
        metavar="N",
        help="output level of the model",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="l1_loss",
        metavar="LOSS",
        help="define the loss function",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="sum",
        metavar="RED",
        help="define the reduction function",
    )
    parser.add_argument(
        "--loss_weights_order",
        type=str,
        default="inc",
        metavar="ORDER",
        help="define the order of the loss weights",
    )
    parser.add_argument(
        "--loss_J",
        type=str,
        default="abs",
        metavar="LOSS_J",
        help="define the jacobian loss function",
    )
    parser.add_argument(
        "--loss_J_gamma",
        type=float,
        default=0.1,
        metavar="LOSS_J_GAMMA",
        help="define the jacobian loss weight",
    )
    parser.add_argument(
        "--full_res",
        type=bool,
        default=False,
        metavar="FULL_RES",
        help="flag to output full-res interpolated output as final resolution",
    )
    parser.add_argument(
        "--full_res_loss_weight_multiplier",
        type=float,
        default=4.0,
        metavar="FULL_RES_WEIGHT",
        help="full-res loss weight multiplier",
    )
    # Logging
    parser.add_argument(
        "--project",
        type=str,
        default="test",
        metavar="P",
        help="project name",
    )
    parser.add_argument("--run", type=str, default=None, metavar="R", help="run name")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        metavar="n",
        help="info output interval for log",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="additional comment for tensorboard"
    )

    # Hardware settings
    parser.add_argument(
        "--seed", type=int, default=5738, metavar="N", help="seed for random generator"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of slave-processes for data loading",
    )
    # Import / export settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="rand_L_lag",
        metavar="T",
        help="training data type",
    )
    return parser

class pykitPIVDataset(Dataset):
    """Load pykitPIV-generated dataset"""

    def __init__(self, image_pairs, targets, transform=None, n_samples=None, pin_to_ram=False):

        self.data = image_pairs
        self.target = targets

        if n_samples:
            self.data = self.data[:n_samples]
            self.target = self.target[:n_samples]
        if pin_to_ram:
            self.data = np.array(self.data)
            self.target = np.array(self.target)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx], self.target[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample

def get_train_test_loader(args):
    # Data loader
    transform = torchvision.transforms.Compose(
        [
            lima.transforms.RandomAffine(
                degrees=17, translate=(0.2, 0.2), scale=(0.9, 2.0)
            ),
            lima.transforms.RandomHorizontalFlip(),
            lima.transforms.RandomVerticalFlip(),
            lima.transforms.ToTensor(),
            lima.transforms.NormalizeBounded(
                bit_depth=8 if args.dataset == "num" else 16
            ),
            lima.transforms.RandomBrightness(factor=(0.5, 2)),
            lima.transforms.RandomNoise(std=(0, args.noise_std)),
        ]
    )
    
    train_dataset = pykitPIVDataset(image_pairs=image_pairs_train,
                                    targets=targets_train,
                                    transform=transform)
    
    test_dataset = pykitPIVDataset(image_pairs=image_pairs_test,
                                    targets=targets_test,
                                    transform=transform)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=5,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=True)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=10)
    
    return train_loader, test_loader


def main(args):
    # 1. Log
    dict_args = vars(args)
    print(dict_args)

    pl.seed_everything(args.seed, workers=True)

    # 2. Setup train/test dataset
    train_loader, test_loader = get_train_test_loader(args)
    args.len_train_loader = len(train_loader)

    # 3. Define the model
    model = lima.LIMA(**dict_args)

    # 4. Define logger
    logger = pl.loggers.WandbLogger(
        project=args.project,
        entity="empa305",
        name=args.run,
        save_dir=os.path.join(os.path.dirname(__file__), "logs"),
        log_model=True,
    )
    logger.watch(model, log="all")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
    )

    # 5. Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator=None if args.num_nodes == 1 else "ddp",
        gpus=1,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
    )

    # 6. Train the model
    trainer.fit(model, train_loader, test_loader)

    # Visualize the prediction:
    image_to_predict = 0
    velocity_component = 0
    predicted_flow = model.inference(torch.from_numpy(image_pairs_test[image_to_predict,:,:,:]).to(dtype=torch.float)).numpy()
    plt.imshow(predicted_flow[velocity_component,:,:], 
               cmap='Blues', 
               origin='lower')
    plt.colorbar()

if __name__ == "__main__":
    parser = argument_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
