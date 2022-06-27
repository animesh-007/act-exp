import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import os
from resnet import *
import torchvision.transforms as transforms



import torch.utils.data
import torchvision.datasets as datasets

import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gen_features(net, dataloader):
    # import pdb; pdb.set_trace()
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (input, target) in enumerate(dataloader):
            inputs = input.to(device)
            targets = target.to(device)
            targets_np = targets.data.cpu().numpy()

            outputs = net(inputs)
            outputs_np = outputs.data.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 100 == 0) or (idx+1 == len(dataloader)):
                print(idx+1, '/', len(dataloader))
            
            # if idx == 300:
            #     break

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, outputs

def tsne_plot(args, save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,f'tsne-{args.name}.png'), bbox_inches='tight')
    print('done!')

def main():
    
    parser = argparse.ArgumentParser(description='Activation Function')

    # Basic arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer type')
    parser.add_argument('--ckpt', type=str, default='./', help='ckpt path')
    parser.add_argument('--act', type=str, default='relu', help='activation type')
    parser.add_argument('--version', type=int, default=20, help='model version')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='For wandb logging')
    parser.add_argument('--name', type=str, default="resnet20",
                        help='For wandb logging')

    
    args = parser.parse_args()
    
    if args.version == 20:
        model = resnet20(act=args.act)
    elif args.version == 32:
        model = resnet32(act=args.act)
    elif args.version == 44:
        model = resnet44(act=args.act)
    elif args.version == 56:
        model = resnet56(act=args.act)

    # import pdb; pdb.set_trace()
    model.load_state_dict(torch.load(args.ckpt, map_location=torch.device(device))["state_dict"])

    model = model.cuda()

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

    def validate(args, val_loader, model):
        """
        Run evaluation
        """
        # batch_time = AverageMeter()
        # losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        # end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda()
            target = target.cuda()

            # compute output
            with torch.no_grad():
                output = model(input)
                # loss = criterion(output, target)

            output = output.float()
            # loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            # losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            if i % 20 == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    # "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    # "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), top1=top1
                    )
                )

        print(" * Prec@1 {top1.avg:.3f}".format(top1=top1))


    prec1 = validate(args, val_loader, model)
    # gen_features(model, val_loader)
    targets, outputs = gen_features(model, val_loader)
    tsne_plot(args, "./tsne", targets, outputs)

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

    


if __name__ == '__main__':
    main()
