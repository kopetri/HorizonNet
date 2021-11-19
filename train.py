import sys
import argparse
import torch
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model import ENCODER_RESNET, ENCODER_DENSENET
from dataset import PanoCorBonDataset
from module import HorizonModel
import numpy as np
from pathlib import Path

def parse_ckpt(path):    
    return [p for p in Path(path).glob("**/*") if p.suffix == ".ckpt"][0].as_posix()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=10, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=300, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')
    parser.add_argument('--find_learning_rate', action='store_true', help="Finding learning rate.")
    parser.add_argument('--detect_anomaly', action='store_true', help='Enables pytorch anomaly detection')

    parser.add_argument('--pth', default=None, help='path to load saved checkpoint. (finetuning)')
    # Model related
    parser.add_argument('--backbone', default='resnet50', choices=ENCODER_RESNET + ENCODER_DENSENET, help='backbone of the network')
    parser.add_argument('--no_rnn', action='store_true', help='whether to remove rnn or not')
    # Dataset related arguments
    parser.add_argument('--train_root_dir', default='data/layoutnet_dataset/train', help='root directory to training dataset. should contains img, label_cor subdirectories')
    parser.add_argument('--valid_root_dir', default='data/layoutnet_dataset/valid', help='root directory to validation dataset. should contains img, label_cor subdirectories')
    parser.add_argument('--no_flip', action='store_true', help='disable left-right flip augmentation')
    parser.add_argument('--no_rotate', action='store_true', help='disable horizontal rotate augmentation')
    parser.add_argument('--no_gamma', action='store_true', help='disable gamma augmentation')
    parser.add_argument('--no_pano_stretch', action='store_true', help='disable pano stretch')
    # optimization related arguments
    parser.add_argument('--freeze_earlier_blocks', default=-1, type=int)
    parser.add_argument('--batch_size', default=4, type=int, help='training mini-batch size')
    parser.add_argument('--optim', default='Adam', help='optimizer to use. only support SGD and Adam')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr_pow', default=0.9, type=float, help='power in poly to drop LR')
    parser.add_argument('--warmup_lr', default=1e-6, type=float, help='starting learning rate for warm up')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='numbers of warmup epochs')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=0, type=float, help='factor for L2 regularization')
    parser.add_argument('--bn_momentum', type=float)
    parser.add_argument('--use_ring_conv', action='store_true', help='Enable ring convolution.')
    parser.add_argument('--name', default='HorizonNet', type=str)

    # testing
    parser.add_argument('--ckpt', default=None, type=str, help="Load checkpoint from version folder")

    args = parser.parse_args()

    if args.detect_anomaly:
        print("Enabling anomaly detection")
        torch.autograd.set_detect_anomaly(True)

    if args.use_ring_conv:
        print("Enable circular convolutions.")
    
    # windows safe
    if sys.platform in ["win32"]:
        args.worker = 0

    # Manage Random Seed
    if args.seed is None: # Generate random seed if none is given
        args.seed = random.randrange(4294967295) # Make sure it's logged
    pl.seed_everything(args.seed)

    callbacks = []

    callbacks += [pl.callbacks.lr_monitor.LearningRateMonitor()]

    callbacks += [pl.callbacks.ModelCheckpoint(
        verbose=True,
        save_top_k=1,
        filename='{epoch}-{valid_3DIoU}',
        monitor='valid_3DIoU',
        mode='max'
    )]

    use_gpu = not args.gpus == 0

    trainer = pl.Trainer(
        log_gpu_memory=False,
        fast_dev_run=args.dev,
        profiler=False,
        gpus=args.gpus,
        log_every_n_steps=1,
        overfit_batches=1 if args.overfit else 0,
        precision=args.precision if use_gpu else 32,
        amp_level='O2' if use_gpu else None,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        logger=pl.loggers.TensorBoardLogger("result", name=args.name),
        callbacks=callbacks
    )

    yaml = args.__dict__
    yaml.update({
            'random_seed': args.seed,
            'gpu_name': torch.cuda.get_device_name(0) if use_gpu else None,
            'gpu_capability': torch.cuda.get_device_capability(0) if use_gpu else None
            })

    train_dataset = PanoCorBonDataset(root_dir=args.train_root_dir, flip=not args.no_flip, rotate=not args.no_rotate, gamma=not args.no_gamma, stretch=not args.no_pano_stretch)
    val_dataset   = PanoCorBonDataset(root_dir=args.valid_root_dir, return_cor=True, flip=False, rotate=False, gamma=False, stretch=False)
    

    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=args.worker,
                              pin_memory=True,
                              worker_init_fn=lambda x: np.random.seed())

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.worker
    )

    # Init variable
    args.warmup_iters = args.warmup_epochs * len(train_loader)
    args.max_iters = args.max_epochs * len(train_loader)
    args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr

    if args.ckpt:
        ckpt = parse_ckpt(args.ckpt)
        print("Loading ckpt: ", ckpt)
        model = HorizonModel.load_from_checkpoint(checkpoint_path=ckpt)
        model.store_ckpt(ckpt)
    else:
        model = HorizonModel(args)


    if args.find_learning_rate:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        suggested_lr = lr_finder.suggestion()
        print("Old learning rate: ", args.learning_rate)
        args.learning_rate = suggested_lr
        print("Suggested learning rate: ", args.learning_rate)
    else:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


   
