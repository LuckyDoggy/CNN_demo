from torch.autograd import Variable
from AverageMeter import AverageMeter
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import random
import models
from dataloader import *
import os

class trainer():
    def __init__(self,args,time_stp):
        self.args=args
        self.time_stp=time_stp
        #从配置文件中加载参数
        with open(self.args.config) as f:
            self.config = yaml.load(f)
        for k, v in self.config['common'].items():
            setattr(self.args, k, v)
        #加载模型
        if args.dataset in ['MNIST', 'FashionMNIST', 'KMNIST']:
            self.model = models.__dict__[args.arch](num_classes=10,data_channels=1)
        elif args.dataset=='CIFAR10':
            self.model = models.__dict__[args.arch](num_classes=10, data_channels=3)
        elif args.dataset=='CIFAR100':
            self.model = models.__dict__[args.arch](num_classes=100, data_channels=3)
        # if "model" in self.config.keys():
        #     self.model = models.__dict__[args.arch](**self.config['model'])
        # else:
        #     self.model = models.__dict__[args.arch]()
        print(self.model)
        #设置训练的device
        self.device = torch.device('cuda:' + str(args.gpus[0]) if torch.cuda.is_available() else "cpu")

    def train(self):

        #设置random seed
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        random.seed(self.args.random_seed)

        #设置GPU参数
        if self.args.use_gpu and torch.cuda.is_available():
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(self.args.random_seed)
            self.args.gpus = [int(i) for i in self.args.gpus.split(',')]
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpus)
            self.model.to(self.device)

        #设置loss和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        #加载数据
        train_loader,test_loader=get_loader(self.args)

        #training
        best_acc=0.0
        for epoch in range(self.args.epochs):
            self.adjust_learning_rate(optimizer, epoch)
            self.train_epoch(train_loader=train_loader,model=self.model,criterion=criterion,optimizer=optimizer,epoch=epoch,device=self.device)

            acc=self.test(test_loader,self.model,epoch,self.device,self.time_stp)
            is_best = acc > best_acc
            if is_best:
                print('epoch: {} The best is {} last best is {}'.format(epoch, acc, best_acc))
            best_acc = max(acc, best_acc)
            if not os.path.exists(self.args.save_path):
                os.mkdir(self.args.save_path)
            save_name = '{}/{}_{}_best.pth.tar'.format(self.args.save_path, self.args.arch, epoch) if is_best else \
                '{}/{}_{}.pth.tar'.format(self.args.save_path, self.args.arch, epoch)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, filename=save_name)



    def train_epoch(self,train_loader,model,criterion,optimizer,epoch,device):
        losses=AverageMeter()
        acc=AverageMeter()

        #模型切换到训练模式
        model.train()

        for i, (input, target) in enumerate(train_loader):

            input_var = Variable(input).float().to(device)
            target_var = Variable(target).long().to(device)
            output = model(input_var)
            loss = criterion(output, target_var)
            reduced_loss = loss.data.clone()
            losses.update(reduced_loss,input.size()[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]['lr']


            with open('./logs/{}_{}.txt'.format(self.time_stp, self.args.arch), 'a+') as flog:
                line = 'Epoch: [{0}][{1}/{2}]\t lr:{3:.5f}\t' \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    .format(epoch, i, len(train_loader), lr,
                            loss=losses,)
                print(line)
                flog.write('{}\n'.format(line))

    def save_checkpoint(self,state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def adjust_learning_rate(self,optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.args.lr * (0.1 ** (epoch // self.args.every_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def test(self,test_loader,model,epoch,device,time_stp):
        #模型切换到测试模式
        model.eval()

        #测试
        T=0
        count=0
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                with torch.no_grad():
                    input_var = Variable(input).float().to(device)
                    count += input_var.size()[0]
                    target_var = Variable(target).long().to(device)
                    output = model(input_var)
                    soft_output = torch.softmax(output, dim=-1)
                    preds = soft_output.to('cpu').detach().numpy()
                    label = target.to('cpu').detach().numpy()
                    _, predicted = torch.max(soft_output.data, 1)
                    predicted = predicted.to('cpu').detach().numpy()
                    T+=sum(np.equal(predicted, label))
        acc=float(T)/count
        #保存测试结果
        with open('logs/val_result_{}_{}_{}.txt'.format(time_stp,self.args.arch,self.args.dataset),'a+') as f_result:
            result_line = 'epoch: {}  Acc:{:.3f}'.format(epoch,acc)
            f_result.write('{}\n'.format(result_line))
            print(result_line)
        return acc