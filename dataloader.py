import pathlib
import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms

import transforms

class Dataset:
    def __init__(self, args):
        self.args=args
        dataset_rootdir = './Dataset'
        self.dataset_dir = dataset_rootdir+'/'+args.dataset
        self._train_transforms = []
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

    def get_datasets(self):
        #提取dataset
        train_dataset = getattr(torchvision.datasets, self.args.dataset)(
            self.dataset_dir,
            train=True,
            transform=self.train_transform,
            download=True)
        test_dataset = getattr(torchvision.datasets, self.args.dataset)(
            self.dataset_dir,
            train=False,
            transform=self.test_transform,
            download=True)
        return train_dataset, test_dataset


    def _get_train_transform(self):
        #对训练数据进行一些预处理操作
        if self.args.arch=='lenet':
            self._train_transforms.append(torchvision.transforms.Resize(32))
        if self.args.use_horizontal_flip:
            self._add_horizontal_flip()
        self._add_normalization()
        self._train_transforms.append(transforms.ToTensor())
        return torchvision.transforms.Compose(self._train_transforms)

    def _get_test_transform(self):
        #对测试集数据的预处理
        test_transforms=[]
        if self.args.arch=='lenet':
            test_transforms.append(torchvision.transforms.Resize(32))
        test_transforms.append(transforms.Normalize(self.mean, self.std))
        test_transforms.append(transforms.ToTensor())
        transform = torchvision.transforms.Compose(test_transforms)
        return transform

    def _add_random_crop(self):
        #添加随机剪切
        transform = torchvision.transforms.RandomCrop(
            self.size, padding=self.args.random_crop_padding)
        self._train_transforms.append(transform)

    def _add_horizontal_flip(self):
        #添加随机水平翻转
        self._train_transforms.append(
            torchvision.transforms.RandomHorizontalFlip())

    def _add_normalization(self):
        #添加标准化
        self._train_transforms.append(
                transforms.Normalize(self.mean, self.std))


class CIFAR(Dataset):
    def __init__(self, args):
        self.size = 32
        dataset_name = args.dataset
        if dataset_name == 'CIFAR10':
            self.mean = np.array([0.4914, 0.4822, 0.4465])
            self.std = np.array([0.2470, 0.2435, 0.2616])
        elif dataset_name == 'CIFAR100':
            self.mean = np.array([0.5071, 0.4865, 0.4409])
            self.std = np.array([0.2673, 0.2564, 0.2762])
        super(CIFAR, self).__init__(args)


class MNIST(Dataset):
    def __init__(self, args):
        self.size = 28
        dataset_name=args.dataset
        if dataset_name == 'MNIST':
            self.mean = np.array([0.1307])
            self.std = np.array([0.3081])
        elif dataset_name == 'FashionMNIST':
            self.mean = np.array([0.2860])
            self.std = np.array([0.3530])
        elif dataset_name == 'KMNIST':
            self.mean = np.array([0.1904])
            self.std = np.array([0.3475])
        super(MNIST, self).__init__(args)




def get_loader(args):
    #提取dataloader对象
    batch_size = args.batch_size
    num_workers = args.num_workers
    use_gpu = args.use_gpu

    dataset_name = args.dataset
    assert dataset_name in [
        'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'KMNIST'
    ]

    if dataset_name in ['CIFAR10', 'CIFAR100']:
        dataset = CIFAR(args)
    elif dataset_name in ['MNIST', 'FashionMNIST', 'KMNIST']:
        dataset = MNIST(args)

    train_dataset, test_dataset = dataset.get_datasets()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader

