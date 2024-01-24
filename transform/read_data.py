from torch.utils.data import Dataset
from PIL import Image
import os

class Mydata(Dataset):#Mydata类继承自Dataset类
    def __init__(self,root_dir,label_dir):#定义类的初始化函数
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(root_dir,label_dir)#将路径拼接起来
        self.image_path=os.listdir(self.path)


    def __getitem__(self, idx):
        img_name=self.image_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)#用Image读取图片
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.image_path)

root_dir= "../dataset/train"
ants_label_dir="ants"
bees_label_dir="bees"
bees_dataset=Mydata(root_dir,bees_label_dir)
ants_dataset=Mydata(root_dir,ants_label_dir)
train_dataset=ants_dataset+bees_dataset

img,label=ants_dataset[0]
img.show()