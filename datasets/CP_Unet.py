import math
import os
import random
import time

import numpy as np
from torch.utils.data import Dataset
import nibabel
import scipy.ndimage as ndi
from scipy import ndimage
import torch
import glob

class CPDataset(Dataset):

    def __init__(self, root_dir, img_list, sets, with_mask=True, with_crop=False):
        self.with_mask = with_mask
        self.with_crop = with_crop
        root_dir = '*npy_data_path*'
        if sets.phase =='train':
            root_dir = '**/**/**'
            item_dirs = glob.glob(os.path.join(root_dir, '*/*.npy'))
            item_dirs = [item for item in item_dirs if '_seg' not in item]
        elif sets.phase =='test':
            root_dir = '**/**/**'
            item_dirs = glob.glob(os.path.join(root_dir, '*/*.npy'))
            item_dirs = [item for item in item_dirs if '_seg' not in item]
        self.item_dirs = item_dirs
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def __nii2tensorarray__(self, data):
        [z, y] = data.shape
        new_data = np.reshape(data, [1, z, y])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        return len(self.item_dirs)

    def __getitem__(self, idx):
        
       
        # read image and labels
        # print('in getitem')

        
        img_name = self.item_dirs[idx]
        label_name = os.path.join(os.path.dirname(img_name), os.path.basename(img_name).split('.')[0]+'_seg.npy')
        # print(img_name,label_name)
        assert os.path.isfile(img_name)
        assert os.path.isfile(label_name)
        img = np.load(img_name)  
        assert img is not None
        mask = np.load(label_name)
        assert mask is not None
        if 'Yes' in img_name:
            label = 1
            ID = int(img_name.split('Yes/')[1].split('_')[0])+1000
        elif 'No' in img_name:
            label = 0
            ID = int(img_name.split('No/')[1].split('_')[0])
        
        # data processing
        if self.phase == 'train':
            img_all_array, mask_all_array, img_crop_array, mask_crop_array = self.__training_data_process__(img, mask)

            # 2 tensor array
            img_all_array = self.__nii2tensorarray__(img_all_array)
            mask_all_array = self.__nii2tensorarray__(mask_all_array)
            # img_crop_array = self.__nii2tensorarray__(img_crop_array)
            # mask_crop_array = self.__nii2tensorarray__(mask_crop_array)

            assert img_all_array.shape ==  mask_all_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_all_array.shape, mask_all_array.shape)
            # assert img_crop_array.shape ==  mask_crop_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_crop_array.shape, mask_crop_array.shape)
            
            # print('===========in dataset =======================')
            # print(img_all_array)
            # print('===========in dataset =======================')
            if self.with_mask and not self.with_crop:
                return img_all_array, mask_all_array,label,ID
            elif self.with_mask and self.with_crop:
                return img_all_array, mask_all_array, img_crop_array, mask_crop_array
            elif not self.with_mask and self.with_crop:
                return img_all_array, img_crop_array
            else:
                return img_all_array
        # elif self.phase == 'test':
        #     img_all_array = self.__testing_data_process__(img)
        #     img_all_array = self.__nii2tensorarray__(img_all_array)
        #     return img_all_array,label,ID
        elif self.phase == 'test':
            img_all_array,mask_all_array = self.__testing_data_process__(img,mask)
            img_all_array = self.__nii2tensorarray__(img_all_array)
            mask_all_array = self.__nii2tensorarray__(mask_all_array)
            assert img_all_array.shape ==  mask_all_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_all_array.shape, mask_all_array.shape)
            return img_all_array,mask_all_array,label,ID


    def __itensity_normalize_one_volume__(self, volume):

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        # out_random = np.random.normal(0, 1, size = volume.shape)
        # out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        [depth, height] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data


    def __crop_data__(self, data, label):
        data_all, label_all, data_crop, label_crop = self.__random_center_crop__ (data, label)
        
        return data_all, label_all, data_crop, label_crop
    


    def __transform_matrix_offset_center__(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def __apply_transform__(self, x,
                        transform_matrix,
                        channel_axis=0,
                        fill_mode='nearest',
                        cval=0.):
        
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        x = ndi.interpolation.affine_transform(
            x,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=cval) 

        return x

    def __random_transform__(self, volume,label=None,rota=15,wsh=0.1,hsh=0.1,inten=0.2,zoom=(0.8,0.8),
                            row_axis=0, col_axis=1, channel_axis=2,fill_mode='nearest', cval=0.):
        h, w = volume.shape[row_axis], volume.shape[col_axis]
        # 对比度增强
        gamma = np.random.uniform(0.7, 1)
        smin = volume.min()
        smax = volume.max()
        sdata = (volume-smin)/(smax-smin)
        contrast_data = np.power(sdata,gamma)
        volume = contrast_data*(smax-smin)+smin
        
        # 旋转
        theta = np.pi / 180 * np.random.uniform(-rota, rota)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
        transform_matrix = rotation_matrix
        
        # 平移
        tx = np.random.uniform(-hsh, hsh) * h
        ty = np.random.uniform(-wsh, wsh) * w
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        transform_matrix = np.dot(transform_matrix, shift_matrix)
        
        # 扭曲
        shear = np.random.uniform(-inten, inten)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                [0, np.cos(shear), 0],
                                [0, 0, 1]])
        transform_matrix = np.dot(transform_matrix, shear_matrix)

        # 缩放
        zoom_range=zoom
        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = np.dot(transform_matrix, zoom_matrix)
        transform_matrix= self.__transform_matrix_offset_center__(transform_matrix, h, w)
        volume = self.__apply_transform__(volume, transform_matrix, channel_axis, fill_mode, cval)
        if label is not None:
            label = self.__apply_transform__(label, transform_matrix, channel_axis, fill_mode, cval)
            return volume,label
        else:
            return volume

    def __training_data_process__(self, data, label): 

        # data_all, label_all, data_crop, label_crop = self.__crop_data__(data, label) 
        data_all = data
        label_all = label

        data_all = self.__resize_data__(data_all)
        label_all = self.__resize_data__(label_all)
        # data_crop = self.__resize_data__(data_crop)
        # label_crop = self.__resize_data__(label_crop)
        data_crop = None
        label_crop = None

        #########################################random transform
        data_all,label_all = self.__random_transform__(data_all,label_all,rota=15,wsh=0.1,hsh=0.1,inten=0.2,zoom=(0.9,0.8))


        # normalization datas
        data_all = self.__itensity_normalize_one_volume__(data_all)
        # data_crop = self.__itensity_normalize_one_volume__(data_crop)

        return data_all, label_all, data_crop, label_crop

    def __contrast__(self,data):
        smin = data.min()
        smax = data.max()
        sdata = (data-smin)/(smax-smin)
        contrast_data = np.power(sdata,0.85)
        rdata = contrast_data*(smax-smin)+smin
        return rdata
    def __testing_data_process__(self, data,mask): 
        # print("testing dataloader")
       
        # resize data
        data = self.__resize_data__(data)
        mask = self.__resize_data__(mask)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)
        data = self.__contrast__(data)

        return data,mask
