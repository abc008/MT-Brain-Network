import numpy as np
import torch
import torch.nn as nn
from setting import parse_opts
import torchvision
import torch.nn.functional as F
from torchvision.models import resnet18


class MyResModel2D(nn.Module):
    def __init__(self, opt):
        super(MyResModel2D,self).__init__()
        self.inplanes = 64
        self.myresnet = resnet18(pretrained=False, progress=True, num_classes=opt.n_seg_classes)
        self.myresnet.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.myresnet.fc = nn.Linear(512, 16)
        
        resnet18_pre = torch.load("./trails/pretrain/resnet18-5c106cde.pth")        
        ignore_pre = ['conv1.weight', 'fc.weight', 'fc.bias']
        state = self.myresnet.state_dict()
        for k in resnet18_pre:
            if k in ignore_pre:
                continue
            else:
                state[k] = resnet18_pre[k]
                # print(k)
        self.myresnet.load_state_dict(state)
        self.dp = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, opt.n_seg_classes)
        
        
    def forward(self,x_3D):
        x_out = self.myresnet(x_3D[:,:,:,:,0])
        for i in range(1,x_3D.shape[4]):
            x_slice = self.myresnet(x_3D[:,:,:,:,i])
            x_out = torch.cat([x_out,x_slice],1)
        # print(x_out.shape)
        x_out = self.dp(x_out)
        x_out = self.fc1(x_out)
        # print(x_out.shape)
        
        return x_out
def resnet18_2D(opt):
    model = MyResModel2D(opt)
    return model
        


if __name__ == "__main__":
    sets = parse_opts()  
    model = MyResModel2D(sets)
    print(model)
    # for k in model.state_dict():
        # print(k) 
    # print(model.state_dict())


