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

class CPDataset(Dataset):

    def __init__(self, root_dir, img_list, sets, with_mask=True, with_crop=False):
        self.with_mask = with_mask
        self.with_crop = with_crop
        root_dir = '*npy_data_path*'
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
            self.img_list = [line.replace('.nii.gz', '.npy') for line in self.img_list]
        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        

        ith_info = self.img_list[idx].split(" ")
        img_name = os.path.join(self.root_dir, ith_info[0])
        label_name = os.path.join(self.root_dir, ith_info[1])
        label = int(ith_info[2])
        if ith_info[0].split('/')[1]=='Yes':
            ID = int((ith_info[0].split('/')[2]).split('.')[0])+1000
        else:
            ID = int((ith_info[0].split('/')[2]).split('.')[0])
        assert os.path.isfile(img_name)
        assert os.path.isfile(label_name)
        img = np.load(img_name)  
        assert img is not None
        mask = np.load(label_name)
        assert mask is not None
        age = int(ith_info[6])
        edema = int(ith_info[3])

        if self.phase == 'train':
            img_all_array, mask_all_array, img_crop_array, mask_crop_array = self.__training_data_process__(img, mask)

            # 2 tensor array
            img_all_array = self.__nii2tensorarray__(img_all_array)
            mask_all_array = self.__nii2tensorarray__(mask_all_array)
            img_crop_array = self.__nii2tensorarray__(img_crop_array)
            mask_crop_array = self.__nii2tensorarray__(mask_crop_array)

            assert img_all_array.shape ==  mask_all_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_all_array.shape, mask_all_array.shape)
            assert img_crop_array.shape ==  mask_crop_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_crop_array.shape, mask_crop_array.shape)
            
            if self.with_mask and not self.with_crop:
                return img_all_array, mask_all_array, label, ID
            elif self.with_mask and self.with_crop:
                return img_all_array, mask_all_array, img_crop_array, mask_crop_array, label, ID
            elif not self.with_mask and self.with_crop:
                return img_all_array, img_crop_array, label, ID
            else:
                return img_all_array, label, ID
        elif self.phase == 'test':
            img_all_array,mask_all_array = self.__testing_data_process__(img,mask)
            img_all_array = self.__nii2tensorarray__(img_all_array)
            mask_all_array = self.__nii2tensorarray__(mask_all_array)
            return img_all_array,mask_all_array,label,ID


        
        
            

    def __drop_invalid_range__(self, volume, label=None):

        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)
        
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        
        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]


    def __random_center_crop__(self, data, label):
        from random import random
        target_indexs = np.where(label>0)
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        
        return data[:,:,min_W: max_W+1], label[:,:,min_W: max_W+1], data[:,:,min_W: max_W+1], label[:,:,min_W: max_W+1]



    def __itensity_normalize_one_volume__(self, volume):
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std

        return out

    def __resize_data__(self, data):
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=3)

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
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
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

        data_all, label_all, data_crop, label_crop = self.__crop_data__(data, label) 

        data_all = self.__resize_data__(data_all)
        label_all = self.__resize_data__(label_all)
        data_crop = self.__resize_data__(data_crop)
        label_crop = self.__resize_data__(label_crop)

        #########################################random transform
        data_all,label_all = self.__random_transform__(data_all,label_all,rota=15,wsh=0.1,hsh=0.1,inten=0.2,zoom=(0.9,1))


        # normalization datas
        data_all = self.__itensity_normalize_one_volume__(data_all)
        data_crop = self.__itensity_normalize_one_volume__(data_crop)

        return data_all, label_all, data_crop, label_crop

    def __contrast__(self,data):
        smin = data.min()
        smax = data.max()
        sdata = (data-smin)/(smax-smin)
        contrast_data = np.power(sdata,0.85)
        rdata = contrast_data*(smax-smin)+smin
        return rdata
    def __crop__(self,data,label):
        target_indexs = np.where(label>0)
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)        
        return data[:,:,min_W: max_W+1], label[:,:,min_W: max_W+1]

    def __testing_data_process__(self, data,mask): 
        # print("testing dataloader")
        data,mask = self.__crop__(data,mask)
        # resize data
        data = self.__resize_data__(data)
        mask= self.__resize_data__(mask)
        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)
        data = self.__contrast__(data)

        return data,mask
