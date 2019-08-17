# CNN_demo
CNN的演示实验，数据集包括CIFAR和MNIST，网络包括lenet,resnet,vgg

- # train lenet on MNIST
  - python main.py --config='./cfgs/lenet_MNIST.yaml'
  
- # train lenet on CIFAR100
  - python main.py --config='./cfgs/lenet_CIFAR100.yaml'
  
- # train resnet18 on MNIST
  - python main.py --config='./cfgs/resnet_MNIST.yaml'
  
- # train resnet18 on CIFAR100
  - python main.py --config='./cfgs/resnet_CIFAR100.yaml'
  
- # train vgg11bn on MNIST
  - python main.py --config='./cfgs/vgg11bn_MNIST.yaml'
  
- # train vgg11bn on CIFAR100
  - python main.py --config='./cfgs/vgg11bn_CIFAR100.yaml'
