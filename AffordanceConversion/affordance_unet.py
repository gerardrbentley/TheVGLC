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

def get_normed_map(affordance, full_affordance_map, opt_affordance_map=None):
    """
    Gets affordance map normalized to range 0.0 - 1.0 and converted to 3 channel image for compatibility
    
    :param: affordance string name of affordance to get map for
    :param: full_affordance_map a tensor of shape (Batch x 9 x Height x Width)
    :param: opt_affordance_map a tensor of shape (Batch x 9 x Height x Width) or none

    :returns: tensor of shape (Batch x 3 x Height x Width). Returns tuple of two tensors if opt_affordance_map is used
    """
    split_affordances = torch.unbind(full_affordance_map, dim=1)
    single_map = split_affordances[AFFORDANCES.index(affordance)]
    single_map = torch.stack((single_map, single_map, single_map), dim=1)
    single_map = TT.img_norm(single_map, range=(0.0, 1.0))
    if opt_affordance_map is None:
        return single_map
    else:
        other_affordances = torch.unbind(opt_affordance_map, dim=1)
        other_map = other_affordances[AFFORDANCES.index(affordance)]
        other_map = torch.stack((other_map, other_map, other_map), dim=1)
        other_map = TT.img_norm(other_map, range=(0.0, 1.0))
        return single_map, other_map

    

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
        if self.hparams.solid_only:
            self.net = UNetAuto(num_out_channels=1, max_features=512)
        else:
            self.net = UNetAuto(num_out_channels=len(AFFORDANCES), max_features=512)

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
        if batch_idx == 0 and self.trainer.root_gpu is not None and self.trainer.root_gpu == predictions.get_device():
            self.log_validation_images(images, predictions, targets)
        to_log = {'val_loss': loss_val}
        output = {
            'val_loss': loss_val,
            'progress_bar': to_log,
        }
        return output

    def validation_epoch_end(self, val_step_outputs):
        # log.info('Val end')
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
        if not self.hparams.solid_only:
            # Add a training image and it's target to logs
            viz_idxs = torch.randint(0, len(self.dataset), (8,)).tolist()

            input_images = []
            target_list = []
            for idx in viz_idxs:
                image, target = self.dataset[idx]
                input_images.append(TT.img_norm(image))
                target_list.append(target)
            viz_inputs = torch.stack(input_images)
            targets = torch.stack(target_list)
            target_solid = get_normed_map('solid', targets)
            target_danger = get_normed_map('dangerous', targets)

            img_grid = make_grid(torch.cat([viz_inputs, target_solid, target_danger]), nrow=len(viz_idxs), padding=20)
            # log.info(f"image grid shape : {img_grid.shape}, {type(img_grid)}")
            pil_grid = to_pil_image(img_grid)
            with tempfile.NamedTemporaryFile(prefix='sample_', suffix='.png') as filepath:
                pil_grid.save(filepath)
                self.logger.experiment.log_artifact(self.logger.run_id, filepath.name, 'pre_training')

    @rank_zero_only
    def log_validation_images(self, inputs, predictions, targets):
        # curr_device = next(self.net.parameters()).get_device()
        # viz_idxs = list(range(8))

        # input_images = []
        # target_list = []
        # for idx in viz_idxs:
        #     image, target = self.val_dataset[idx]
        #     input_images.append(image)
        #     target_list.append(target)
        # model_inputs = torch.stack(input_images, dim=0)
        # input_images = [TT.img_norm(image) for image in input_images]
        # viz_inputs = torch.stack(input_images, dim=0)

        # model_outputs = self.forward(model_inputs.to(curr_device))
        # model_outputs = model_outputs.cpu()
        
        # targets = torch.stack(target_list, dim=0)
        # pred_solid, target_solid = get_normed_map('solid', model_outputs, targets)
        # pred_danger, target_danger = get_normed_map('dangerous', model_outputs, targets)

        # img_grid = make_grid(torch.cat([viz_inputs, pred_solid, target_solid, pred_danger, target_danger]), nrow=len(viz_idxs), padding=20)
        viz_inputs = TT.img_norm(inputs)
        model_outputs = TT.img_norm(predictions, range=(0.0,1.0))
        viz_targets = TT.img_norm(targets, range=(0.0,1.0))
        # pri /nt(viz_inputs.shape, model_outputs.shape, viz_targets.shape)
        model_outputs = torch.cat((model_outputs, model_outputs, model_outputs), dim=1)
        viz_targets = torch.cat((viz_targets, viz_targets, viz_targets), dim=1)
        # print(viz_inputs.shape, model_outputs.shape, viz_targets.shape)
        
        img_grid = make_grid(torch.cat([viz_inputs, model_outputs, viz_targets]), nrow=inputs.shape[0], padding=20)
        pil_grid = to_pil_image(img_grid.cpu())
        # log.info(img_grid.shape, torch.cat(viz_inputs).shape)
        filename = f"globalstep_{self.global_step:05d}_sample_outputs"
        with tempfile.NamedTemporaryFile(prefix=filename, suffix='.png') as filepath:
            pil_grid.save(filepath)
            self.logger.experiment.log_artifact(self.logger.run_id, filepath.name, 'eval_images')

    #Default mean and std are from super mario bros images
    def configure_transforms(self, mean=TT.DEFAULT_MEAN, std=TT.DEFAULT_STD):
        # train_flag = not args.no_augmentation
        train_flag = False
        train_transform = TT.get_transform(
            train=train_flag, mean=mean, std=std)
        val_transform = TT.get_transform(
            train=False, mean=mean, std=std)
        return train_transform, val_transform

    def prepare_data(self):
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
        # if self.use_ddp:
        #     self.train_sampler = torch.utils.data.distributed.DistributedSampler(
        #         self.dataset)
        #     self.val_sampler = torch.utils.data.distributed.DistributedSampler(
        #         self.val_dataset)
        # else:
        #     self.train_sampler = torch.utils.data.RandomSampler(self.dataset)
        #     self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)


    @pl.data_loader
    def train_dataloader(self):
        """
        Required
        """
        log.info('Training data loader called.')
        # self.load_full_dataset()
        return torch.utils.data.DataLoader(self.dataset, num_workers=16, 
            batch_size=self.hparams.batch_size, drop_last=True)

    @pl.data_loader
    def val_dataloader(self):
        """
        Optional
        """
        log.info('Validation data loader called.')
        # self.load_full_dataset()
        return torch.utils.data.DataLoader(self.val_dataset, num_workers=16, 
            batch_size=self.hparams.batch_size, drop_last=True)

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
                        save_top_k=2,
                        verbose=True,
                        monitor='val_loss',
                        mode='min',
                        prefix=args.model
                    )

    trainer = pl.Trainer(
        logger=logger,
        gpus=use_gpu,
        distributed_backend=distributed_backend,
        num_nodes=args.world_size,
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=args.accumulations,
        fast_dev_run=args.fast_run,
        profiler=True,
        weights_summary='top',
        max_epochs=100
    )
    trainer.fit(model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
