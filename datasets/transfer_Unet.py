import os
import glob
import tqdm
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


def crop(data, label):

    target_indexs = np.where(label>0)
    [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
    [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
    # print(min_W,max_W)
    
    return data[:,:,min_W: max_W+1], label[:,:,min_W: max_W+1]

# 为了加速读取及图片处理过程，可以提前将.nii文件转为.npy
root_dir = '/root_data_path/**/**/'
dst_root = '/save_data_path/**/**/'
img_list = '/**/**/**/train_or_test_list.txt'
with open(img_list, 'r') as f:
    img_list = [line.strip() for line in f]
    img_list = [line.replace('.nii.gz', '.npy') for line in img_list]
    

for src_item in tqdm.tqdm(img_list, ascii=True):
    # print(src_item)
    ith_info = src_item.split(" ")
    img_name = os.path.join(root_dir, ith_info[0])
    label_name = os.path.join(root_dir, ith_info[1])
    image_data = np.load(img_name)
    label_data = np.load(label_name)
    image_data,label_data = crop(image_data,label_data)

    img_name = img_name.replace('test', 'train') 
    label_name = label_name.replace('test', 'train') 
    # img_name = img_name.replace('train', 'test') 
    # label_name = label_name.replace('train', 'test') 

    
    
    for i in range(image_data.shape[2]):
        new_name = img_name.split('.')[0]+'_{}'.format(i)+ '.npy'
        new_mask_name = label_name.split('_')[0]+'_{}'.format(i)+ '_seg.npy'
        # print(new_name,new_mask_name)

        new_name = 'train'+new_name.split('train')[1]
        new_mask_name = 'train'+new_mask_name.split('train')[1]
        # new_name = 'test'+new_name.split('test')[1]
        # new_mask_name = 'test'+new_mask_name.split('test')[1]

        dst_item = os.path.join(dst_root, new_name)
        dst_mask_item = os.path.join(dst_root, new_mask_name)
        if os.path.exists(dst_item):
            continue
        data = image_data[:,:,i]
        mask = label_data[:,:,i]
        np.save(dst_item, data)
        np.save(dst_mask_item, mask)
        

    
        
    