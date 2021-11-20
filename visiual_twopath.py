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
    # print(module) # 用于区分模块
    # print('input', len(input), input[0].shape) # 首先打印出来
    fmap_block.append(output)
def hook_fn_backward(module, grad_in, grad_out):
    # print('backward',grad_out[0].shape)
    grad_block.append(grad_out[0].detach())

def comp_class_vec(ouput_vec, index=None):
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    # add new dimensions (increase dimensions) to the NumPy array
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 2).scatter_(1, index, 1)
    print(one_hot)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot.cuda() * ouput_vec)  

    return class_vec

def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128, 128))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam

def show_cam_on_image(img, mask,batch,num, out_dir,contours):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # print(heatmap.shape)
    # print(img.shape)
    # print(mask[90])
    # print(255 * img[90])
    cam = heatmap* 0.15  + img* 0.3
    cam = cam / np.max(cam)
    cam_counter = cam
    cam = cam.transpose((1,0,2))
    cam_counter = cam_counter.transpose((1,0,2))
    img = img.transpose((1,0,2))

    
    path_cam_img = os.path.join(out_dir, "cam{}_{}.png".format(batch,num))
    path_raw_img = os.path.join(out_dir, "raw{}_{}.png".format(batch,num))
    path_ged_img = os.path.join(out_dir, "ged{}_{}.png".format(batch,num))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # plt.imshow(np.uint8(255*mask),cmap='jet')
    # plt.colorbar()
    # fig = plt.gcf()
    # plt.margins(0,0)
    # fig.savefig(path_ged_img,dpi=500,bbox_inches="tight")
    # plt.close("all")
    # # print(np.uint8(255 * cam)[90])

    cam_plt = cam[:, :, ::-1]
    plt.imshow(np.uint8(255*cam_plt),cmap='jet')
    # plt.colorbar()
    fig = plt.gcf()
    plt.margins(0,0)
    plt.axis('off')
    fig.savefig(path_ged_img,dpi=500,bbox_inches="tight")
    plt.close("all")

    


if __name__ == '__main__':
    PATH = './trails/models/trained_parm**.tar'
    classes = ('未侵袭', '侵袭')

    # 存储feature map 和 权重
    fmap_block = []
    grad_block = []


    sets = parse_opts()
    tmp = ''
    for id in sets.gpu_id:
        tmp += str(id)+','
   
    os.environ["CUDA_VISIBLE_DEVICES"]= tmp
        
    # getting model
    torch.manual_seed(sets.manual_seed)
    torch.cuda.manual_seed_all(sets.manual_seed)
    model, parameters = generate_model(sets) 
    # print(model)
    # 1/0

    model = model.cuda() 
    model.eval()
    modules = model.myRes2D[0].myresnet.layer4[1].named_children() # 
    for name, module in modules:
        if name == 'conv2':
            module.register_forward_hook(hook_fn_forward)
            module.register_backward_hook(hook_fn_backward)
            print(name, module)
    # 1/0

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(PATH)['state_dict'])
    


    # input = torch.rand((1, 1, 128, 128, 32)).cuda()
    sets.phase = 'test'
    testing_dataset = CPDataset(sets.data_root, '/data/**/data_list.txt', sets)
    test_loader = DataLoader(testing_dataset, batch_size=4, shuffle=False, num_workers=sets.num_workers, pin_memory=True)
    print("=================")
    for batch_id, batch_data in enumerate(test_loader):
        data, label,right_idx,ID = batch_data
        
        # data_batch = next(iter(test_loader))[0]
        print(data.shape,ID)

        for i in range(data.shape[0]):
            # forward
            output,output_mask = model(data[i,:,:,:,:].unsqueeze(0), data[i,:,:,:,:].unsqueeze(0))
            # output = model(data[i,:,:,:,:].unsqueeze(0), data[i,:,:,:,:].unsqueeze(0))
            idx = np.argmax(output.cpu().data.numpy())
            # print(idx)
            print("predict: {}".format(classes[idx]))
            print("right: {}".format(classes[right_idx[i]]))

            # backward
            model.zero_grad()
            class_loss = comp_class_vec(output,idx)
            class_loss.backward(retain_graph=True)

            
            # CAM
            for j in range(data.shape[4]):
            # for j in range(1):
                # print(len(grad_block))
                grads_val = grad_block[i*32+j][0].cpu().data.numpy().squeeze()
                fmap = fmap_block[i*32+j][0].cpu().data.numpy().squeeze()
                cam = gen_cam(fmap, grads_val)
                # print(cam.shape)

                # 保存cam图片
                data_cam = np.expand_dims(data[i,0,:,:,j], axis=2)
                data_cam = np.concatenate((data_cam, data_cam, data_cam), axis=-1)
                # 归一化到0-1
                data_cam = (data_cam-np.min(data_cam))/(np.max(data_cam)-np.min(data_cam))
                # print(data_cam.min(),data_cam.max())
                # 寻找轮廓边界
                # print(label.shape)
                label_cam = label[i,0,:,:,j].cpu().detach().numpy()
                label_cam=label_cam.astype(np.uint8) 
                # print(label_cam.shape)
                # print(np.max(label_cam))
                contours,_ = cv2.findContours(label_cam,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_show = np.float32(cv2.resize(np.float32(data_cam), (128, 128)))
                show_cam_on_image(img_show, cam,ID[i],j, './cam/twopath_mask_unco',contours)
            # 1/0

    # print(len(total_feat_in))
    print(len(fmap_block))
    print(len(grad_block))
