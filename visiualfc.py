import torch
import cv2
import os
import torch.nn as nn
from model import generate_model
from datasets.CP_npy import CPDataset 
from setting import parse_opts 
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt 
from torch.autograd import Variable
import numpy as np
import nibabel
from scipy import ndimage



def hook_fn_forward(module, input, output):
    print("======Modile======",module) # 用于区分模块
    print('======input=======', len(input), input[0].shape) # 首先打印出来
    fmap_block.append(input)
def draw_features(height,width,x,out_dir):
    savename= os.path.join(out_dir, "fc.pdf")
    # print(savename)
    x = x.cpu().detach().numpy()
    pmin = np.min(x)
    pmax = np.max(x)
    fig = plt.figure()

    # plt.axis('off')
    img = x
    img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
    # print(img)
    img=img.astype(np.uint8)  #转成unit8
    # print(img)
    img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
    img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
    plt.imshow(img)
    plt.xlabel("Features")
    plt.ylabel('Patients')
    fig = plt.gcf()
    fig.savefig(savename, dpi=800,pad_inches=0.0,bbox_inches="tight")
    fig.clf()
    plt.close()


if __name__ == '__main__':
    PATH = './trails/models/trained_param**.tar'
    classes = ('未侵袭', '侵袭')

    # 存储feature map 和 权重
    fmap_block = []

    sets = parse_opts()
    tmp = ''
    for id in sets.gpu_id:
        tmp += str(id)+','
   
    os.environ["CUDA_VISIBLE_DEVICES"]= tmp
        
    # getting model
    torch.manual_seed(sets.manual_seed)
    torch.cuda.manual_seed_all(sets.manual_seed)
    # cudnn.deterministic = True
    model, parameters = generate_model(sets) 
    # print(model)

    model = model.cuda() 
    model.eval()
    # modules = model.myRes2D[0].layer1[1].named_children() # 
    # for name, module in modules:
    #     if name == 'relu':
    #         module.register_forward_hook(hook_fn_forward)
    #         print(name, module)
    modules = model.named_children() # 
    for name, module in modules:
        if name == 'fc2':
            module.register_forward_hook(hook_fn_forward)
            # module.register_backward_hook(hook_fn_backward)
            print(name, module)

        
    
    
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(PATH)['state_dict'])
    


    # input = torch.rand((1, 1, 128, 128, 32)).cuda()
    sets.phase = 'test'
    testing_dataset = CPDataset(sets.data_root, '/data/path**/**.txt', sets)
    test_loader = DataLoader(testing_dataset, batch_size=16, shuffle=False, pin_memory=True)

    for batch_id, batch_data in enumerate(test_loader):
        data,mask, inva,ID = batch_data
        if not sets.no_cuda: 
            data = data.cuda()
        # for i in range(data.shape[0]):
        #     print(data.shape[0])
        # forward
        output,out_mask = model(data, data)
        idx = np.argmax(output.cpu().data.numpy(),1)
        print("output shape",output.shape)
        print("idx",idx)
        # print("length of fmap_block",len(fmap_block))
        # print(idx)
        # print("predict: {}".format(classes[idx]))

    # print("total  length of fmap_block",len(fmap_block))
    fmap = fmap_block[0][0].cpu()
    # print("fmap",fmap)
    for i in range(1,len(fmap_block)):
        fmap = torch.cat((fmap,fmap_block[i][0].cpu()),0)
    
    print(fmap.shape)
    draw_features(fmap.shape[0],512,fmap,'./FC')
        

