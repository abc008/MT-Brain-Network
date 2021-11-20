import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import manifold
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
    fmap_block.append(input)

def draw_tsne(feature,label):

    X = feature.detach().numpy()
    y = label.detach().numpy()
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print(X.shape)
    print(X_tsne.shape)
    print(y.shape)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    vis_x = X_norm[:,0]
    vis_y = X_norm[:,1]
    plt.scatter(vis_x[:128],vis_y[:128],c='blue',s=100,label='$Non-invasion$',marker='.',edgecolors= 'white')
    plt.legend()
    ax = plt.gca() 
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(3)
    print(y[127],y[128])
    plt.scatter(vis_x[128:],vis_y[128:],c='red',s=100,label='$Invasion$',marker='*',edgecolors= 'white')
    plt.legend(facecolor='None')
    plt.xticks([])
    plt.yticks([])

    plt.show()
    # plt.title('t-SNE Embedding of Classification Results')
    fig = plt.gcf()
    fig.savefig('./FC/tsne.pdf',dpi=800,bbox_inches="tight",pad_inches=0.0)
    plt.close("all")
    



if __name__ == '__main__':
    PATH = './trails/models/**/trained_parm.tar'
    classes = ('未侵袭', '侵袭')

    # 存储feature map 和 权重
    fmap_block = []

    sets = parse_opts()
    tmp = ''
    for id in sets.gpu_id:
        tmp += str(id)+','
   
    os.environ["CUDA_VISIBLE_DEVICES"]= tmp
        
    torch.manual_seed(sets.manual_seed)
    torch.cuda.manual_seed_all(sets.manual_seed)
    model, parameters = generate_model(sets) 

    model = model.cuda() 
    model.eval()
    modules = model.named_children() # 
    for name, module in modules:
        if name == 'fc2':
            module.register_forward_hook(hook_fn_forward)
            # module.register_backward_hook(hook_fn_backward)
            print(name, module)

        
    
    
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(PATH)['state_dict'])
    


    sets.phase = 'test'
    testing_dataset = CPDataset(sets.data_root, '/data/**/data_list**.txt', sets)
    test_loader = DataLoader(testing_dataset, batch_size=16, shuffle=False, pin_memory=True)
    labellist = []
    for batch_id, batch_data in enumerate(test_loader):
        data,mask, inva,ID = batch_data
        if not sets.no_cuda: 
            data = data.cuda()
        # for i in range(data.shape[0]):
        #     print(data.shape[0])
        # forward
        # twopath+coord+mask
        output,out_mask = model(data, data)
        # twopath_only
        # output = model(data, data)
        idx = np.argmax(output.cpu().data.numpy(),1)
        labellist.append(inva)


    fmap = fmap_block[0][0].cpu()
    # print("fmap",fmap)
    for i in range(1,len(fmap_block)):
        fmap = torch.cat((fmap,fmap_block[i][0].cpu()),0)
    label = torch.cat(labellist).cpu() 
    print(fmap.shape)
    print(label.shape)
    # print(label)
    draw_tsne(fmap,label)