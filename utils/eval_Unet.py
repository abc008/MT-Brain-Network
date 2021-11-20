import os
import random
import sys
import torch
from skimage.morphology import remove_small_objects,binary_opening
import numpy as np

def average(list):
    s = 0
    for item in list:
        s += item
    return s/len(list)

def sum(list):
    s = 0
    for item in list:
        s += item
    return s

def analysis(x,y):
    '''
    对输入的两个四维张量[B,1,H,W]进行逐图的DSC、PPV、Sensitivity计算
    其中x表示网络输出的预测值
    y表示实际的预想结果mask
    返回为一个batch中DSC、PPV、Sen的平均值及batch大小
    '''
    
    x = torch.from_numpy(x.astype(np.uint8)).cuda()
    # y = torch.from_numpy(y.astype(np.uint8))
    x=x.type(dtype=torch.uint8)
    y=y.type(dtype=torch.uint8)#保证类型为uint8
    DSC=[]
    PPV=[]
    Sen=[]
    if x.shape==y.shape:
        batch=x.shape[0]
        for i in range(batch):#按第一个维度分开
            
            tmp = torch.eq(x[i],y[i])
            
            tp=int(torch.sum(torch.mul(x[i]==1,tmp==1))) #真阳性
            fp=int(torch.sum(torch.mul(x[i]==1,tmp==0))) #假阳性
            fn=int(torch.sum(torch.mul(x[i]==0,tmp==0))) #假阴性
        
        
            try:
                DSC.append(2*tp/(fp+2*tp+fn))
            except:
                DSC.append(0)
            try:
                PPV.append(tp/(tp+fp))
            except:
                PPV.append(0)
            try:
                Sen.append(tp/(tp+fn))
            except:
                Sen.append(0)
            
                
    else:
        sys.stderr.write('Analysis input dimension error')
        

    DSC = sum(DSC)/batch
    PPV = sum(PPV)/batch
    Sen = sum(Sen)/batch
    return DSC, PPV, Sen, batch


def post_process(img,min_size=100):
    '''
    图像后处理过程
    包括开运算和去除过小体素
    返回uint16格式numpy二值数组
    '''
    img = img.cpu()
    img = img.numpy().astype(np.bool)
    b,c,w,h = img.shape
    if c==1:
        for i in range(b):
            img_tmp = img[i,0,:,:]
            img_tmp = binary_opening(img_tmp)
            remove_small_objects(img_tmp, min_size=min_size, in_place=True)
            img_tmp = ~remove_small_objects(~img_tmp, min_size=min_size)
            img[i,0,:,:] = img_tmp
        
    return img.astype(np.uint16)