"""
Based on template from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/lightning_module_template.py

Adapted by Gerard
"""
import logging as log
import os
import json
import time
import datetime
import tempfile
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

import pytorch_lightning as pl
from pytorch_lightning.logging import MLFlowLogger

import segmentation_transforms as TT
from models import get_model, UNetAuto
from ml_args import parse_args

from datasets import GameLevelsDataset, get_dataset, dataset_mean_std, get_stats

AFFORDANCES = ["changeable", "dangerous", "destroyable", "gettable", "movable", 
        "portal", "solid", "ui", "usable"]

def rank_zero_only(fn):
    """Decorate a method to run it only on the process with logger rank 0.

    :param fn: Function to decorate
    """

    @wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if self.logger is not None and self.logger.rank == 0:
            fn(self, *args, **kwargs)

    return wrapped_fn

class AffordanceUnetLightning(pl.LightningModule):
    """
    Unet Auto Encoder models training with pytorch lightning
    """

    def __init__(self, hparams):
        """
        Pass in parsed hyperparams to the model, initialize Lighning superclass
        :param hparams:
        """
        super(AffordanceUnetLightning, self).__init__()
        self.hparams = hparams
        self.dataset = None
        self.net = UNetAuto(num_out_channels=len(AFFORDANCES), max_features=1024)

    def forward(self, x):
        """
        Based on model choice. Could include Linear and Convolutional Layers here
        :param x:
        :return:
        """
        return self.net.forward(x)

    # ---------------------
    # TRAINING
    # ---------------------
    def loss(self, targets, segmentations):
        """
        Combines sigmoid with Binary Cross Entropy for numerical stability.
        Can include pos_weight to increase recall of underrepresented classes
        :param targets: ground truth image or segmentation map
        :param segmentations: output of the autoencoder network
        """
        bce_loss = F.binary_cross_entropy_with_logits(segmentations, targets)
        return bce_loss

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return: dict
            - loss -> tensor scalar [REQUIRED]
            - progress_bar -> Dict for progress bar display. Must have only tensors
            - log -> Dict of metrics to add to logger. Must have only tensors (no images, etc)
        """
        # forward pass
        images, targets = batch
        predictions = self.forward(images)
        # calculate loss
        loss_val = self.loss(targets, predictions)
        to_log = {'training_loss': loss_val}
        output = {
            'loss': loss_val,  # required
            'progress_bar': to_log,
            'log': to_log,
        }
        return output
    
    # TODO investigate full batch loss. Lets all distributed batches come together easily
    # def training_end(self, train_step_outputs):

    @rank_zero_only
    def on_epoch_end(self):
        curr_device = next(self.net.parameters()).get_device()
        viz_idxs = [0,1,2]
        viz_inputs = []
        for idx in viz_idxs:
            image, target = self.val_dataset[idx]
            image, target = image.unsqueeze(0), target.unsqueeze(0)
            model_output = self.forward(image.to(curr_device))

            image = TT.img_norm(image)
            viz_inputs.append(image)

            all_affordances = torch.unbind(model_output, dim=1)
            solid_map = all_affordances[AFFORDANCES.index('solid')]
            solid_map = torch.stack((solid_map, solid_map, solid_map), dim=1)
            solid_map = TT.img_norm(solid_map)
            viz_inputs.append(solid_map.cpu())
            # log.info(f"solid map: {solid_map.min()}, {solid_map.max()}, {torch.unique(solid_map)}")
            # viz_inputs.append(target)

        img_grid = make_grid(torch.cat(viz_inputs), nrow=2, padding=40)
        pil_grid = to_pil_image(img_grid)
        # log.info(img_grid.shape, torch.cat(viz_inputs).shape)
        filename = f"globalstep_{self.global_step:05d}_sample_outputs"
        with tempfile.NamedTemporaryFile(prefix=filename, suffix='.png') as filepath:
            pil_grid.save(filepath)
            self.logger.experiment.log_artifact(self.logger.run_id, filepath.name, 'eval_images')


    @rank_zero_only
    def on_train_end(self):
        args = self.hparams
        total_time = time.time() - self.start_time
        total_seconds = int(total_time)
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        log.info('Training time {}'.format(total_time_str))
        param_dict = {
            'epochs': args.epochs,
            'num_samples': len(self.dataset),
            'batch_size': args.batch_size,
            'lr_start': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'distributed': 'None' if args.gpus == '' else args.backend,
            'gpus': 'None' if args.gpus == '' else args.gpus,
            'world_size': args.world_size,
            'train_time': total_time_str,
            'train_seconds': total_seconds,
        }

        log.info(json.dumps(param_dict, indent=2, sort_keys=True))
        for key, val in param_dict.items():
            self.logger.experiment.log_param(self.logger.run_id, key, val)

    # ---------------------
    # VALIDATION LOOP
    # ---------------------
    
    def validation_step(self, batch, batch_idx):
        """
        Called in validation loop with model in eval mode
        """
        images, targets = batch
        predictions = self.forward(images)
        loss_val = self.loss(targets, predictions)
        to_log = {'val_loss': loss_val}
        output = {
            'val_loss': loss_val,
            'progress_bar': to_log,
        }
        return output

    def validation_end(self, val_step_outputs):
        log.info('Val end')
        val_losses = torch.stack([x['val_loss'] for x in val_step_outputs])
        
        avg_loss = val_losses.mean()
        max_loss = val_losses.max()
        to_log = {'val_loss': avg_loss,
                'max_val_loss': max_loss}
        output = {
            'val_loss':avg_loss,
            'log': to_log
        }
        return output
        

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        REQUIRED
        can return multiple optimizers and learning_rate schedulers
        Use lists to return lr scheduler, else can just return optimizer
        :return: list of optimizers
        """
        args = self.hparams
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #     factor=0.1)
        return optimizer

    @rank_zero_only
    def on_train_start(self):
        self.start_time = time.time()
        # Add a training image and it's target to tensorboard
        rand_select = torch.randint(0, len(self.dataset), (6,)).tolist()
        train_images = []
        for idx in rand_select:
            data = self.dataset[idx]
            image, target = data.image, data.target

            train_images.append(image)
            all_affordances = torch.unbind(target, dim=0)
            solid_map = all_affordances[AFFORDANCES.index('solid')]
            solid_map = torch.stack((solid_map, solid_map, solid_map), dim=0)
            solid_map = TT.img_norm(solid_map)
            train_images.append(solid_map.float())
            # train_images.append(target)

        img_grid = make_grid(
            train_images, nrow=6, padding=20, normalize=True)
        log.info(f"image grid shape : {img_grid.shape}, {type(img_grid)}")
        pil_grid = to_pil_image(img_grid)
        with tempfile.NamedTemporaryFile(prefix='sample_', suffix='.png') as filepath:
            pil_grid.save(filepath)
            self.logger.experiment.log_artifact(self.logger.run_id, filepath.name, 'pre_training')

    #Default mean and std are from super mario bros images
    def configure_transforms(self, mean=TT.DEFAULT_MEAN, std=TT.DEFAULT_STD):
        # train_flag = not args.no_augmentation
        train_flag = False
        train_transform = TT.get_transform(
            train=train_flag, mean=mean, std=std)
        val_transform = TT.get_transform(
            train=False, mean=mean, std=std)
        return train_transform, val_transform

    def load_full_dataset(self):
        if self.dataset is None:
            # Dataset Mean and Std
            if self.hparams.dataset_mean:
                log.info('Calculating dataset mean')
                temp_ds = get_dataset(
                    self.hparams.dataset, transform=TT.ToTensor())
                self.mean, self.std = dataset_mean_std(temp_ds)
            else:
                # Fetch stats based on dataset, defaults to megaman if not found
                self.mean, self.std = get_stats(self.hparams.dataset)
            log.info(f"Dataset mean: {self.mean}, std: {self.std}")
            # Dataset Augmentations
            train_transform, val_transform = self.configure_transforms(mean=self.mean, std=self.std)

            self.dataset = get_dataset(
                self.hparams.dataset, transform=train_transform)

            self.val_dataset = get_dataset(
                self.hparams.dataset, transform=val_transform)

            num_samples = len(self.dataset)
            num_train = int(0.8 * num_samples)
            indices = torch.randperm(num_samples).tolist()
            self.dataset = torch.utils.data.Subset(
                self.dataset, indices[0:num_train])
            self.val_dataset = torch.utils.data.Subset(
                self.val_dataset, indices[num_train:])
            log.info(f'Full Dataset loaded: len: {num_samples}')
            # self.train_split, self.val_split = torch.utils.data.random_split(
            #     self.dataset, (num_train, num_samples - num_train))
            log.info(
                f'Train {len(self.dataset)} and Val {len(self.val_dataset)} splits made')

            # Distributed Data Parallel mode chunks the dataset so that each worker does equal work but doesn't do extra work
            if self.use_ddp:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.dataset)
                self.val_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.val_dataset)
            else:
                self.train_sampler = torch.utils.data.RandomSampler(self.dataset)
                self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)


    @pl.data_loader
    def train_dataloader(self):
        """
        Required
        """
        log.info('Training data loader called.')
        self.load_full_dataset()
        return torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, 
            batch_size=self.hparams.batch_size, drop_last=True)

    @pl.data_loader
    def val_dataloader(self):
        """
        Optional
        """
        log.info('Validation data loader called.')
        self.load_full_dataset()
        return torch.utils.data.DataLoader(self.val_dataset, sampler=self.val_sampler, 
            batch_size=self.hparams.batch_size, drop_last=False)

    # @pl.data_loader
    # def test_dataloader(self):
    #     log.info('Test data loader called.')
    #     return self.__dataloader(train=False)


def main(args):
    # init module
    model = AffordanceUnetLightning(args)

    use_gpu = None if args.gpus == '' else args.gpus
    distributed_backend = None if args.gpus == '' else args.backend
    
    if args.do_log:
        #TODO non ultra64 tracking use https://ultra64.cs.pomona.edu/mlflow/
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'username'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
        mlflow_tracking_uri = 'file:///faim/mlflow/mlruns/'
        logger = MLFlowLogger(experiment_name=args.experiment, tracking_uri=mlflow_tracking_uri)
        # Instantiates experiment if it didn't exist
        run_id = logger.run_id
        exp_id = logger.experiment.get_experiment_by_name(args.experiment).experiment_id
        artifact_location = f'/faim/mlflow/mlruns/{exp_id}/{run_id}/artifacts'
    else:
        artifact_location = './artifacts'
        logger = False
    log.info(f'saving artifacts to {artifact_location}')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                        filepath=artifact_location,
                        save_top_k=1,
                        verbose=True,
                        monitor='val_loss',
                        mode='min',
                        prefix=args.model
                    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        gpus=use_gpu,
        distributed_backend=distributed_backend,
        num_nodes=args.world_size,
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=args.accumulations,
        fast_dev_run=args.fast_run,
    )
    trainer.fit(model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
