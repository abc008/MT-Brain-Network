import numpy as np
import torch
import torch.nn as nn
from setting import parse_opts
import torchvision
import torch.nn.functional as F
from torchvision.models import resnet18
from models.coordconv import CoordConv2d

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable
    
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode=='deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels = out_channels,

                                                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode=='pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        if self.BN_enable:
            self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        # print("==================")
        # print(x.shape)
        x=self.conv(x)
        if self.BN_enable:
            x=self.norm1(x)
        x=self.relu1(x)
        x=self.upsample(x)
        if self.BN_enable:
            x=self.norm2(x)
        x=self.relu2(x)
        return x

class MyResModel2DUnetOnly(nn.Module):
    def __init__(self, opt,BN_enable=True):
        super(MyResModel2DUnetOnly,self).__init__()
                
        print('unet_only_coord_4')

        self.BN_enable=BN_enable
        self.inplanes = 64
        self.coordconv = CoordConv2d(1,32,1,with_r=True)
        filters=[64,64,128,256,512]
        self.myresnet = resnet18(pretrained=False, progress=True, num_classes=opt.n_seg_classes)
        self.myresnet.conv1 = nn.Conv2d(32, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.myresnet.fc = nn.Linear(512, 16)
        self.coordconv2 = CoordConv2d(256,256,1,with_r=True)
        resnet18_pre = torch.load("./trails/pretrain/resnet18-5c106cde.pth",)        
        ignore_pre = ['conv1.weight', 'fc.weight', 'fc.bias']
        state = self.myresnet.state_dict()
        for k in resnet18_pre:
            if k in ignore_pre: 
                continue
            else:
                if 'layer4' not in k:
                    state[k] = resnet18_pre[k]
                # print(k)
        self.myresnet.load_state_dict(state)
        # print("======================")


        # decoder部分
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3]*4, out_channels=filters[3], BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[3]+filters[2], mid_channels=filters[2]*4, out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[2]+filters[1], mid_channels=filters[1]*4, out_channels=filters[1], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[1]+filters[0], mid_channels=filters[0]*4, out_channels=filters[0], BN_enable=self.BN_enable)
        if self.BN_enable:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), 
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                # nn.Sigmoid()
                )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1), 
                # nn.Sigmoid()
                )
        self.sig = nn.Sigmoid()
        
    def forward(self,x_3D):
        # print("================================myresnet2D.shape=======================")
        # print(x_3D.shape)
        # print("================================myresnet2D.shape=======================")
        x = self.coordconv(x_3D)
        x = self.myresnet.conv1(x)
        x = self.myresnet.bn1(x)
        x = self.myresnet.relu(x)
        # print('x.shape',x.shape)
        x_ = self.myresnet.maxpool(x)
        # print('x_.shape',x_.shape)
        e1 = self.myresnet.layer1(x_)
        # print('e1.shape',e1.shape)
        e2 = self.myresnet.layer2(e1)
        # print('e2.shape',e2.shape)
        e3 = self.myresnet.layer3(e2)
        # print('e3.shape',e3.shape)

        x_out = self.coordconv2(e3)
        x_out = self.myresnet.layer4(x_out)
        x_out = self.myresnet.avgpool(x_out)
        x_out = torch.flatten(x_out, 1)
        x_out = self.myresnet.fc(x_out)


        center = self.center(e3)
        # print('center.shape',center.shape)

        d2 = self.decoder1(torch.cat([center,e2],dim=1))
        # print('d2.shape',d2.shape)
        d3 = self.decoder2(torch.cat([d2,e1],dim=1))
        # print('d3.shape',d3.shape)
        d4 = self.decoder3(torch.cat([d3,x],dim=1))
        # print('d4.shape',d4.shape)
        x_att = self.final(d4)
        # print('maskpred.shape',x_mask.shape)
        # print('out.shape',x_out.shape)
        # print(x_out,x_mask)
        x_mask = self.sig(x_att)
        return x_out,x_mask,x_att
def resnet18_2DUnetOnly(opt):
    model = MyResModel2DUnetOnly(opt)
    return model
        


if __name__ == "__main__":
    sets = parse_opts()  
    model = MyResModel2DUnetOnly(sets)
    print(model)
    # for k in model.state_dict():
        # print(k) 
    # print(model.state_dict())


