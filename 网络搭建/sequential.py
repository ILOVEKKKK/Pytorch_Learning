import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
writer=SummaryWriter("logs")
class my_net(nn.Module):
    def __init__(self):
        super(my_net,self).__init__()
        self.conv1=nn.Conv2d(3,32,5,1,padding=2)
        self.max_pool1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(32,32,5,padding=2)
        self.max_pool2=nn.MaxPool2d(2)
        self.conv3=nn.Conv2d(32,64,5,padding=2)
        self.max_pool3=nn.MaxPool2d(2)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(1024,64)
        self.linear2=nn.Linear(64,10)
        self.model1=nn.Sequential(#简化代码结构
            self.conv1,
            self.max_pool1,
            self.conv2,
            self.max_pool2,
            self.conv3,
            self.max_pool3,
            self.flatten,
            self.linear1,
            self.linear2
        )

    def forward(self,input):
        output=self.model1(input)
        return output

net=my_net()
x=torch.ones(64,3,32,32)
output1=net(x)
print(output1.shape)
writer.add_graph(net,x)
writer.close()
