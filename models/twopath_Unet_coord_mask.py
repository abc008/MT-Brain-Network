import torch
from models.resnet3D import ResNet,BasicBlock
from models.resnet3D import resnet10
from models.unet_only_coord_4 import resnet18_2DUnetOnly
import torch.nn as nn
from setting import parse_opts

        
class TwoPath(nn.Module):
    def __init__(self,opt):
        super(TwoPath,self).__init__()
        self.myRes3D = resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        self.myRes2D = resnet18_2DUnetOnly(opt)
        
        if not opt.no_cuda:
            self.myRes3D = self.myRes3D.cuda() 
            self.myRes3D = nn.DataParallel(self.myRes3D, device_ids=None)
            self.myRes2D = self.myRes2D.cuda() 
            self.myRes2D = nn.DataParallel(self.myRes2D, device_ids=None)
        res3D_pre = torch.load("./trails/models/**/pretrained_Resnet3D**.tar")
        
        
        self.myRes3D.load_state_dict(res3D_pre['state_dict'])
        
        # for name,p2 in self.myRes3D.named_parameters():  
            # print(name,p2.shape,p2.requires_grad)
        # print("====================Res3D======================")
        # print(list(self.myRes3D.children()))
        self.myRes3D = nn.Sequential(*list(self.myRes3D.module.children())[:-1])
        
        
        
        resnet18_Unet = torch.load('./trails/models//**/pretrained_Resnet2D**.tar')
        # 选择是否需要梯度
        ignore_pre = ['module.myresnet.fc.weight', 'module.myresnet.fc.bias']
        for name,p2 in self.myRes2D.named_parameters():
            
            if name not in ignore_pre and 'layer4' not in name:
                p2.requires_grad = False
            # print(name,p2.shape,p2.requires_grad)
        # # # # 选择加载预训练权重
        # ignore_pre = ['module.myresnet.fc.weight', 'module.myresnet.fc.bias']
        state = self.myRes2D.state_dict()
        for k in resnet18_Unet['state_dict']:
            if k in ignore_pre or 'layer4'in k:
                continue
            else:
                state[k] = resnet18_Unet['state_dict'][k]
                # print(k)
        self.myRes2D.load_state_dict(state)
        # self.myresnet2D.load_state_dict(resnet18_Unet['state_dict'],strict=False)
        # print("====================Res2D======================")
        # print(list(self.myRes2D.children()))
        self.myRes2D = nn.Sequential(*list(self.myRes2D.children()))
        self.dp1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024,512)
        self.dp2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,opt.n_seg_classes)
        




    def forward(self,x_3D,x_2D):

        x_2D_all,x_mask,x_att = self.myRes2D(x_2D[:,:,:,:,0])
        x_mask = x_mask.unsqueeze(-1)
        x_att = x_att.unsqueeze(-1)
        for i in range(1,x_2D.shape[4]):
            x_slice_out,x_slice_mask,x_slice_att = self.myRes2D(x_2D[:,:,:,:,i])
            x_slice_mask = x_slice_mask.unsqueeze(-1)
            x_slice_att = x_slice_att.unsqueeze(-1)
            x_2D_all = torch.cat([x_2D_all,x_slice_out],1)
            x_mask = torch.cat([x_mask,x_slice_mask],4)
            x_att = torch.cat([x_att,x_slice_att],4)

        att = x_att +1
        x_3D = torch.mul(x_3D,att)
        x_3D = self.myRes3D(x_3D)
        x_3D = torch.flatten(x_3D, 1)

        x_out = torch.cat([x_3D,x_2D_all],1)
        x_out = self.dp1(x_out)
        x_out = self.fc1(x_out)
        x_out = self.dp2(x_out)
        x_out = self.fc2(x_out)

        return x_out,x_mask



if __name__ == '__main__':
    sets = parse_opts()  
    model = TwoPath(sets)
    print(model)
    for name, parameters in model.named_parameters():#打印出每一层的参数的大小
       print(name, ':', parameters.size())
        
