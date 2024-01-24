import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("E:/P_project/pytorch_learn/torchvision数据集/dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
data_loader=DataLoader(dataset,batch_size=64,shuffle=True)
writer=SummaryWriter("logs")

class my_net(nn.Module):
    def __init__(self):
        super(my_net,self).__init__()
        self.conv1=nn.Conv2d(3,6,(3,3),stride=(1,1))
        self.max_pool=nn.MaxPool2d(kernel_size=(3,3),stride=3,ceil_mode=False)
        self.relu=nn.ReLU(inplace=False)

    def forward(self,input):
        input=self.conv1(input)
        input=self.max_pool(input)
        output=self.relu(input)
        return output

net=my_net()
step=0

for data in data_loader:
    imgs,labels=data
    output=net(imgs)
    output=torch.reshape(output,(-1,3,10,10))
    #print(output.shape)
    writer.add_images("after_net",output,1)
    step+=1

writer.close()