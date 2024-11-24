import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src import ff_mnist, ff_model


from abc import ABC

class Getter(ABC):
    def __init__(self):
        super(Getter, self).__init__()
    
    def get_model_and_optimizer(opt):
        model = ff_model.FF_model(opt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == opt.device:
            model = model.cuda()
        else:
            model = model.to(device)

        # Printing the model architecture.
        print(model, "\n")

        main_model_params = [
            p
            for p in model.parameters()
            if all(p is not x for x in model.linear_classifier.parameters())
        ]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": main_model_params,
                    "lr": opt.training.learning_rate,
                    "weight_decay": opt.training.weight_decay,
                    "momentum": opt.training.momentum,
                },
                {
                    "params": model.linear_classifier.parameters(),
                    "lr": opt.training.downstream_learning_rate,
                    "weight_decay": opt.training.downstream_weight_decay,
                    "momentum": opt.training.momentum,
                },
            ]
        )
        return model, optimizer

    def get_data(opt, partition):
        dataset = ff_mnist.FF_MNIST(opt, partition)

        dataloader = DataLoader(
            dataset,
            batch_size=opt.input.batch_size,
            drop_last=True,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(opt.seed),
            num_workers=4,
            persistent_workers=True,
        )

        return dataloader

    def get_MNIST_partition(opt, partition):
        if partition in ["train", "val", "train_val"]:
            mnist = torchvision.datasets.MNIST(
                os.path.join(get_original_cwd(), opt.input.path),
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
        elif partition in ["test"]:
            mnist = torchvision.datasets.MNIST(
                os.path.join(get_original_cwd(), opt.input.path),
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
        else:
            raise NotImplementedError

        if partition == "train":
            mnist = torch.utils.data.Subset(mnist, range(50000))
        elif partition == "val":
            mnist = torchvision.datasets.MNIST(
                os.path.join(get_original_cwd(), opt.input.path),
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
            mnist = torch.utils.data.Subset(mnist, range(50000, 60000))

        return mnist

class Logger(ABC):
    def print_results(partition, iteration_time, scalar_outputs, epoch=None):
        if epoch is not None:
            print(f"Epoch {epoch} \t", end="")

        time_str = f"Time: {timedelta(seconds=iteration_time)}"
        scalar_str = " \t".join(f"{key}: {value:.4f}" for key, value in scalar_outputs.items()) if scalar_outputs else ""
        
        print(f"{partition} \t \t{time_str} \t{scalar_str}\n")

    def log_results(res_dict, scalar_outputs, num_steps):
        for key, value in scalar_outputs.items():
            if isinstance(value, float):
                res_dict[key] += value / num_steps
            else:
                res_dict[key] += value.item() / num_steps
        return res_dict

class Helper(ABC):
    def parse_args(opt):
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)

        print(OmegaConf.to_yaml(opt))
        return opt


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def dict_to_cuda(dict):
        for key, value in dict.items():
            dict[key] = value.cuda(non_blocking=True)
        return dict


    def preprocess_inputs(self,opt, inputs, labels):
        if "cuda" in opt.device:
            inputs = self.dict_to_cuda(inputs)
            labels = self.dict_to_cuda(labels)
        return inputs, labels


    def get_linear_cooldown_lr(opt, epoch, lr):
        if epoch > (opt.training.epochs // 2):
            return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
        else:
            return lr


    def update_learning_rate(self,optimizer, opt, epoch):
        optimizer.param_groups[0]["lr"] = self.get_linear_cooldown_lr(
            opt, epoch, opt.training.learning_rate
        )
        optimizer.param_groups[1]["lr"] = self.get_linear_cooldown_lr(
            opt, epoch, opt.training.downstream_learning_rate
        )
        return optimizer


    def get_accuracy(opt, output, target):
        with torch.no_grad():
            prediction = torch.argmax(output, dim=1)
            return (prediction == target).sum() / opt.input.batch_size


