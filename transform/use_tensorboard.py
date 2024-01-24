from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from PIL import Image
root_dir= "../dataset/train/ants"
img_path=os.path.join(root_dir,os.listdir(root_dir)[0])
writer=SummaryWriter("../logs")
img=Image.open(img_path)
img_array=np.array(img)
writer.add_image("test",img_array,1,dataformats="HWC")
for i in range(100):
    writer.add_scalar("test",i,i)
writer.close()
