from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os
#定义tensorboard事件
writer=SummaryWriter("../logs")


#读取数据集的图片
root_dir= "../dataset/train/ants"
img_name=os.listdir(root_dir)
img_path=os.path.join(root_dir,img_name[6])
img=Image.open(img_path)

#将图像转换为tensor类型
tensor_trans=transforms.ToTensor()
img_tensor=tensor_trans(img)

#对数据做归一化
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(img_tensor)

#对图像做resize
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
img_resize=tensor_trans(img_resize)



# 在tensorboard中观察
writer.add_image("Resize",img_resize)
writer.close()


