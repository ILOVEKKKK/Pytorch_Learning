import torch
from torch import nn



class my_net(nn.Module):
    def __init__(self):
        super(my_net,self).__init__()
        self.model=nn.Sequential(
                nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),
                nn.MaxPool2d(2),
                nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2),
                nn.MaxPool2d(2),
                nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1024,64),
                nn.Linear(64,10)
            )

    def forward(self,input):
        output=self.model(input)
        return output


# input=torch.ones(64,3,32,32)
# output=net(input)
# print(output.shape)