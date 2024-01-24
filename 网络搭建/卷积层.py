import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("E:/P_project/pytorch_learn/torchvision数据集/dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader=DataLoader(dataset,batch_size=64)
writer=SummaryWriter("logs")

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=nn.Conv2d(3,6,3,stride=1,padding=0)#RGB图像故输入通道为3

    def  forward(self,x):
        x=self.conv1(x)
        return x

net=Model()#实例化网络对象



step=0
for data in dataloader:
    imgs,labels=data
    output=net(imgs)
    print(output)
    writer.add_images("input",imgs,step)
    output=torch.reshape(output,(-1,3,30,30))#将output变形以在tensorboard中显示
    writer.add_images("output",output,step)
    step+=1

writer.close()