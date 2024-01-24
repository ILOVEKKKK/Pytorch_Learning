import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision

dataset=torchvision.datasets.CIFAR10("../torchvision数据集/dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
data_loader=DataLoader(dataset,64,drop_last=True)

class my_net(nn.Module):
    def __init__(self):
        super(my_net,self).__init__()
        self.linear=nn.Linear(196608,10)

    def forward(self,input):
        output=self.linear(input)
        return output

net=my_net()
step=0

for data in data_loader:
    imgs,labels=data
    # output=torch.reshape(imgs,(1,1,1,-1))
    output=torch.flatten(imgs)
    output=net(output)
    step+=1