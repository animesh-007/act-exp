best_prec1 = 0
evaluate = True

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb
from datetime import datetime
# from ranger import Ranger
from resnet import *
from torch.autograd import Variable
import argparse

import random

seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def main():

    parser = argparse.ArgumentParser(description='Activation Function')

    # Basic arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer type')
    parser.add_argument('--act', type=str, default='relu', help='activation type')
    parser.add_argument('--version', type=int, default=20, help='model version')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='For wandb logging')
    parser.add_argument('--name', type=str, default="resnet20",
                        help='For wandb logging')

                

    args = parser.parse_args()
    

    global best_prec1, evaluate

    args.name = "_".join([args.name, datetime.now().strftime("%b-%d_%H:%M:%S")])

    if args.wandb:
        wandb.init(project="Activation Function", name=args.name)
        wandb.config.update(args)

    if args.version == 20:
        model = resnet20(act=args.act)
    elif args.version == 32:
        model = resnet32(act=args.act)
    elif args.version == 44:
        model = resnet44(act=args.act)
    elif args.version == 56:
        model = resnet56(act=args.act)
    
    if args.wandb:
        wandb.watch(model)
    
    model = model.cuda()

    print(args)

    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    # else:
    #     optimizer = Ranger(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], last_epoch=0 - 1
    )
    max_epoch = 200

    for epoch in range(0, max_epoch):

        print("current lr {:.5e}".format(optimizer.param_groups[0]["lr"]))
        if args.wandb:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % 20 == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                },
                is_best,
                filename=os.path.join("./checkpoints", f"{args.name}_vanilla_checkpoint.th"),
            )

        save_checkpoint(
            {
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
            },
            is_best,
            filename=os.path.join("./checkpoints", f"{args.name}_vanilla_model.th"),
        )

    if args.wandb:
        wandb.run.finish()


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )


def validate(args, val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                )
            )

    print(" * Prec@1 {top1.avg:.3f}".format(top1=top1))

    if args.wandb:
        wandb.log(
            {
                "epoch": epoch,
                "Top-1 accuracy": top1.avg,
                "loss": losses.avg,
            }
        )

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """
    Save the training model
    """
    torch.save(state, filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":

    main()
