import argparse
import json
from collections import OrderedDict
import models

#定义训练和模型的参数
def get_parser():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser = argparse.ArgumentParser(description='PyTorch CNN demo')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='models architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument(
        '--dataset',
        type=str,
        default='MNIST',
        choices=['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'KMNIST'])
    parser.add_argument('--use_horizontal_flip', type=bool,default=True)
    parser.add_argument('--random_crop_padding', type=int, default=4)
    parser.add_argument('--config', default='./cfgs/local_test.yaml')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument("--random_seed", type=int, default=10,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument('--use_gpu', type=bool, default=False, help='whether use gpus training')
    parser.add_argument('--gpus', type=str, default='0', help='use gpus training eg.--gups 0,1')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--val', '--evaluate', dest='evaluate', default=False, type=bool,
                        help='evaluate models on test set')
    parser.add_argument('--val_save', default=False, type=bool,
                        help='whether to save evaluate result')
    parser.add_argument('--every_decay', default=10, type=int, help='how many epoch decay the lr')
    args=parser.parse_args()
    return args



