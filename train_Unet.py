from setting import parse_opts 
from datasets.CP_Unet import CPDataset 
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import utils as vutils
import time
from utils.logger import log
from scipy import ndimage
import os
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, precision_score,recall_score,accuracy_score,precision_recall_curve,average_precision_score
import csv
from test_Unet import test
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def	forward(self, predict, target):
        N = target.size(0)
        smooth = 1
        predict_flat = predict.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = (predict_flat * target_flat).sum(1)
        union = predict_flat.sum(1) + target_flat.sum(1)

        loss = (2* intersection + smooth) / (union + smooth)
        loss = 1 - loss.sum() / N
        
        return loss
        
def train(data_loader,test_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets,writer):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    weights = [1.0,2.0]
    class_weights = torch.FloatTensor(weights).cuda()
    loss_cls = nn.CrossEntropyLoss(weight=class_weights,ignore_index=-1)
    loss_seg = nn.BCELoss()
    loss_seg_dice = DiceLoss()

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()
        loss_cls = loss_cls.cuda()
        loss_seg_dice = loss_seg_dice.cuda()
        
    
    train_time_sp = time.time()
    global_step = 0
    for epoch in range(total_epochs):
        model.train()
        # scheduler.step(epoch)
        labellist = []
        IDlist =[]
        predlist = []
        problist = []


        log.info('Start epoch {}'.format(epoch))
        
        # scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))
        
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes_all,mask_all,label,ID = batch_data
            # volumes_all, masks_all,volumes_crop, masks_crop,label,ID = batch_data
            if not sets.no_cuda: 
                volumes_all = volumes_all.cuda()
                mask_all = mask_all.cuda()
                # volumes_crop = volumes_crop.cuda()


            optimizer.zero_grad()
            if sets.model =="twopath":
                out_label = model(volumes_all,volumes_all)
            elif sets.model =="Unet":
                out_label,out_mask = model(volumes_all)
            else:
                out_label = model(volumes_all)
            # print(out_mask.shape)
            # print(out_mask[0,0,64,:])
            # 1/0
            # print(label)
            
            prob =[F.softmax(el,dim=0) for el in out_label]
            # calculating loss
            label = label.cuda()
            loss_value_cls=loss_cls(out_label,label)
            loss_value_seg=loss_seg(out_mask,mask_all)
            loss_value_seg_dice = loss_seg_dice(out_mask,mask_all)
            loss = loss_value_seg+loss_value_cls+loss_value_seg_dice
            
            loss.backward()    
            # # 查看梯度是否回传
            # for name, parms in model.named_parameters():
	        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))            
            optimizer.step()

            writer.add_scalar('class_loss',loss.item(),global_step=global_step)
            global_step += 1
            
            _, preds = torch.max(out_label, 1)
            # print(preds)
            predlist.append(preds)
            labellist.append(label)
            IDlist.append(ID)
            problist.append(prob)


            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            # log.info(
            #         'Batch: {}-{} ({}), loss = {:.3f}, loss_cls = {:.3f}, avg_batch_time = {:.3f}'\
            #         .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_cls.item(), avg_batch_time))
        # scheduler.step()
        # scheduler.step(epoch)
        
        writer.add_scalars('Lr', {'learning_rate':scheduler.get_last_lr()[0] }, global_step=epoch)

        class_label = torch.cat(labellist).cpu()
        class_ID = torch.cat(IDlist).cpu()
        class_preds = torch.cat(predlist).cpu()
        class_probs = torch.cat([torch.stack(batch) for batch in problist])


        train_ID_label = {}
        train_ID_preds = {}
        csvf = open('./trails/result/trainpreds_{}.csv'.format(epoch),'w')
        fileheader = ['ID','pred']
        dict_writer = csv.DictWriter(csvf,fileheader)
        dict_writer.writerow(dict(zip(fileheader, fileheader)))
        # print(set(test_ID.numpy().tolist()))
        for i_d in set(class_ID.numpy().tolist()):
            idx = (class_ID == i_d).nonzero(as_tuple=False)
            # train_ID_label[i_d]= class_label[idx].squeeze()
            train_ID_preds[i_d]= class_preds[idx].squeeze()
            dict_writer.writerow({"ID": int(i_d), "pred": train_ID_preds[i_d]})
        # print(test_ID_label)
        # print(test_ID_preds)
        csvf.close()

        acc_score = accuracy_score(class_label, class_preds)
        # precision = precision_score(class_label, class_preds, average='weighted')
        # recall = recall_score(class_label, class_preds, average='weighted')
        # print(acc_score)

        # 计算ROC AUC  ROC 第二个参数是1类别神经元对应的输出结果
        class_label_auc = class_label.detach().numpy()
        class_probs_auc = class_probs[:,1].cpu().detach().numpy()
        fpr, tpr, thersholds = roc_curve(class_label_auc,class_probs_auc)
        auc_score = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(class_label_auc,class_probs_auc)
        AP = average_precision_score(class_label_auc,class_probs_auc)

        if (epoch+1)%10==0:
            
            output_img = vutils.make_grid(out_mask[0,:,:,:], padding=0, normalize=True, range=(0,255))
            output_tmp = torch.ge(out_mask, 0.5).mul(255)
            writer.add_scalar('scalar/loss', loss, batch_id)
            writer.add_image('image/input', volumes_all[0,:,:,:])
            writer.add_image('image/mask', mask_all[0,:,:,:])
            writer.add_image('image/predict', output_tmp[0,:,:,:])
            writer.add_image('image/output', out_mask[0,:,:,:])
            plt.subplot(221)
            plt.imshow(volumes_all[0,:,:,:].cpu().detach().numpy().squeeze(), cmap=plt.cm.gray_r)
            plt.subplot(222)
            plt.imshow(mask_all[0,:,:,:].cpu().detach().numpy().squeeze(), cmap=plt.cm.gray_r)
            plt.subplot(223)
            plt.imshow(output_tmp[0,:,:,:].cpu().detach().numpy().squeeze(), cmap=plt.cm.gray_r)
            plt.subplot(224)
            plt.imshow(out_mask[0,:,:,:].cpu().detach().numpy().squeeze(), cmap=plt.cm.gray_r)
            plt.show()
            plt.savefig('./fig/{}_epoch'.format(epoch+1))

            


        avg_epoch_time = (time.time() - train_time_sp) / (1 + epoch)
        log.info(
                'epoch:{}\tloss_seg:{:.3f}\tloss_cls:{:.3f}\tloss_seg_dice:{:.3f}\tacc:{:.3f}\tAUC:{:.3f}\tAP:{:.3f}'\
                .format(epoch, loss_value_seg.item(),loss_value_cls.item(),loss_value_seg_dice.item(),acc_score,auc_score,AP))

        # save model
        if (epoch+1) %2== 0 :
            log.info('Start epoch {}'.format(epoch))
            model_save_path = '{}_epoch_{}pth_3D.tar'.format(save_folder, epoch)
            model_save_dir = os.path.dirname(model_save_path)
            torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        model_save_path)
            train_acc_score,train_auc_score,train_loss=test(data_loader,model,sets,epoch)
            test_acc_score,test_auc_score,test_loss=test(test_loader,model,sets,epoch)
            writer.add_scalars('acc', {'train_acc': train_acc_score, 'test_acc': test_acc_score}, global_step=epoch)
            writer.add_scalars('auc', {'train_auc': train_auc_score, 'test_auc': test_auc_score}, global_step=epoch)

    writer.close()                           
    print('Finished training')            



if __name__ == '__main__':

    # settting
    sets = parse_opts()

    tmp = ''
    for id in sets.gpu_id:
        tmp += str(id)+','
   
    os.environ["CUDA_VISIBLE_DEVICES"]= tmp
     
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 
    
    model = model.cuda() 
    model = nn.DataParallel(model)
    # print (model)
    
    # optimizer
    if sets.model =="twopath":
        params = [
                    { 'params': parameters['base_parameters'], 'lr': sets.learning_rate/100 }, 
                    { 'params': parameters['new_parameters'], 'lr': sets.learning_rate}
                    ]
    else:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,80],gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10,2)
    
    
    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True

    import time
    get_time = time.strftime('%Y-%m-%d-%H-%M')
    log_dir = os.path.join('log', get_time)
    writer = SummaryWriter(log_dir=log_dir)


    training_dataset = CPDataset(sets.data_root, sets.img_list, sets)
    train_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    sets.phase = 'test'
    testing_dataset = CPDataset(sets.data_root, sets.img_list_test, sets)
    test_loader = DataLoader(testing_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    # training
    train(train_loader,test_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets,writer=writer) 