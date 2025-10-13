"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

This file is from Andrej Karpathy's MinGPT.
https://github.com/karpathy/minGPT
"""
import wandb
import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, dev_dataset, labels=None, labels_dev=None, validation_interval=250):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.callbacks = defaultdict(list)
        self.labels = labels # added for CSCI 662 - classification labels
        self.labels_dev = labels_dev # added for CSCI 662 - classification labels
        self.validation_interval = validation_interval

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

        self.wandb_run = wandb.init(
            entity="sabourih-usc",
            project="csci662-fall2024_hw2",
            config={
                "task": config.get("task"),
                "device": config.get("device"),
                "num_workers": config.get("num_workers"),
                "max_iters": config.get("max_iters"),
                "batch_size": config.get("batch_size"),
                "learning_rate": config.get("learning_rate"),
                "betas": config.get("betas"),
                "weight_decay": config.get("weight_decay"),
                "grad_norm_clip": config.get("grad_norm_clip"),
                "input_file": config.get("input_file"),
                "validation_interval": config.get("validation_interval"),
            },
        )

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # dataset splits
        if self.labels is not None:
            # classification
            dataset = torch.utils.data.TensorDataset(self.train_dataset, self.labels)
            dev_dataset = torch.utils.data.TensorDataset(self.dev_dataset, self.labels_dev)
        else:
            # language modeling
            # X is all but last token, Y is all but first token
            X = self.train_dataset[:, :-1]
            Y = self.train_dataset[:, 1:]
            dataset = torch.utils.data.TensorDataset(X, Y)
            X_dev = self.dev_dataset[:, :-1]
            Y_dev = self.dev_dataset[:, 1:]
            dev_dataset = torch.utils.data.TensorDataset(X_dev, Y_dev)

        # setup the dataloader
        train_loader = DataLoader(
            dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        dev_loader = DataLoader(
            dev_dataset,
            sampler=torch.utils.data.SequentialSampler(dev_dataset),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            if self.labels is not None:
                # classification
                self.logits, self.lm_loss, self.classification_logits, self.classification_loss = model(x, classification_targets=y)
                print(self.classification_loss)
                self.wandb_run.log({"classification_loss": self.classification_loss})
            else:
                # language modeling
                self.logits, self.lm_loss, self.classification_logits, self.classification_loss = model(x, targets=y)
                self.wandb_run.log({"lm_loss": self.lm_loss})


            # backprop and update the parameters
            model.zero_grad(set_to_none=True)

            if self.labels is not None:
                # do classification loss
                loss = self.classification_loss
                self.batch_labels = y
                self.wandb_run.log({"classification_loss": self.classification_loss})
            else:
                # do language modeling loss
                loss = self.lm_loss
                self.wandb_run.log({"lm_loss": self.lm_loss})
            
            if self.iter_num % self.validation_interval == 0:
                model.eval()
                total_val_loss = 0.0
                total_val_batches = 0
                with torch.no_grad():
                    for val_batch in dev_loader:
                        val_batch = [t.to(self.device) for t in val_batch]
                        x_val, y_val = val_batch
                        if self.labels is not None:
                            # classification
                            _, _, _, val_classification_loss = model(x_val, classification_targets=y_val)
                            total_val_loss += val_classification_loss.item()
                        else:
                            # language modeling
                            _, val_lm_loss, _, _ = model(x_val, targets=y_val)
                            total_val_loss += val_lm_loss.item()
                        total_val_batches += 1
                avg_val_loss = total_val_loss / total_val_batches
                if self.labels is not None:
                    self.wandb_run.log({"validation_classification_loss": avg_val_loss})
                else:
                    self.wandb_run.log({"validation_lm_loss": avg_val_loss})
                model.train()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
        self.wandb_run.finish()
