import torch
from models.resnet3D import ResNet,BasicBlock
from models.resnet3D import resnet10
from models.resnet2D_only import resnet18_2D
import torch.nn as nn
from setting import parse_opts


        
class TwoPath(nn.Module):
    def __init__(self,opt):
        super(TwoPath,self).__init__()

        print('twopath only')

        self.myRes3D = resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        self.myRes2D = resnet18_2D(opt)
        
        if not opt.no_cuda:
            self.myRes3D = self.myRes3D.cuda() 
            self.myRes3D = nn.DataParallel(self.myRes3D, device_ids=None)
            self.myRes2D = self.myRes2D.cuda() 
            self.myRes2D = nn.DataParallel(self.myRes2D, device_ids=None)

        res3D_pre = torch.load('./trails/models//**/pretrained_Resnet3D**.tar')
        self.myRes3D.load_state_dict(res3D_pre['state_dict'])
        
        self.myRes3D = nn.Sequential(*list(self.myRes3D.module.children())[:-1])
        
        res2D_pre = torch.load('./trails/models//**/pretrained_Resnet2D**.tar')
        self.myRes2D.load_state_dict(res2D_pre['state_dict'])
        self.myRes2D = nn.Sequential(*list(self.myRes2D.module.children())[:-2])
        # for p2 in self.myRes2D.parameters():
        #     p2.requires_grad = False


        self.dp1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024,512)
        self.dp2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,opt.n_seg_classes)
        




    def forward(self,x_3D,x_2D):

        x_3D = self.myRes3D(x_3D)
        x_3D = torch.flatten(x_3D, 1)

        x_2D_all = self.myRes2D(x_2D[:,:,:,:,0])
        for i in range(1,x_2D.shape[4]):
            x_slice = self.myRes2D(x_2D[:,:,:,:,i])
            x_2D_all = torch.cat([x_2D_all,x_slice],1)

        x_out = torch.cat([x_3D,x_2D_all],1)
        x_out = self.dp1(x_out)
        x_out = self.fc1(x_out)
        x_out = self.dp2(x_out)
        x_out = self.fc2(x_out)

        return x_out



if __name__ == '__main__':
    sets = parse_opts()  
    model = TwoPath(sets)
    print(model)
    for name, parameters in model.named_parameters():#打印出每一层的参数的大小
       print(name, ':', parameters.size())
        
