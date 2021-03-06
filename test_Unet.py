from setting import parse_opts 
from datasets.CP_Unet import CPDataset 
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, precision_score,recall_score,accuracy_score,precision_recall_curve,average_precision_score
import csv
from utils.eval_Unet import post_process,analysis
import time

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

def resume(path,model,optimizer):
    if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    return model, optimizer 

def test(testloader, model, sets, epoch):
    model.eval() 
    labellist = []
    out_labellist = []
    IDlist =[]
    predlist = []
    problist = []

    DSC_sum = 0
    PPV_sum = 0
    Sen_sum = 0
    batch_sum = 0
    loss_seg_sum = 0

    weights = [1.0,2.0]
    class_weights = torch.FloatTensor(weights).cpu()
    loss_cls = nn.CrossEntropyLoss(weight=class_weights,ignore_index=-1)
    loss_seg = nn.BCELoss()
    loss_seg_dice = DiceLoss()
    # loss_seg = nn.BCELoss()

    for index,data in enumerate(testloader):
        volumes_all,mask_all,label,ID = data
        # volumes_all, masks_all,volumes_crop, masks_crop,label,ID = data
        if not sets.no_cuda: 
            volumes_all = volumes_all.cuda()
            mask_all = mask_all.cuda()
            # volumes_crop = volumes_crop.cuda()
        
        DSC = 0
        PPV = 0
        Sen = 0
        batch = 0
        with torch.no_grad():
            if sets.model =="twopath":
                out_label = model(volumes_all,volumes_all)
            elif sets.model =="Unet":
                out_label,out_mask = model(volumes_all)
            else:
                out_label = model(volumes_all)
            # out_label = model(volumes_all,volumes_all)

            # loss_seg_vaule = loss_seg(out_mask,mask_all)
            loss_seg_vaule = loss_seg(out_mask,mask_all) + loss_seg_dice(out_mask,mask_all)
            out_mask = torch.ge(out_mask, 0.5).type(dtype=torch.float32) #?????????
            out_mask = post_process(out_mask)
            DSC, PPV, Sen, batch = analysis(out_mask,mask_all)
            DSC_sum += DSC*batch
            PPV_sum += PPV*batch
            Sen_sum += Sen*batch
            batch_sum += batch
            loss_seg_sum += loss_seg_vaule*batch

            prob =[F.softmax(el,dim=0) for el in out_label]

            _, preds = torch.max(out_label, 1)
            out_labellist.append(out_label)
            predlist.append(preds)
            labellist.append(label)
            IDlist.append(ID)
            problist.append(prob)
    DSC_sum /= batch_sum
    PPV_sum /= batch_sum
    Sen_sum /= batch_sum
    loss_seg_sum /= batch_sum


    class_out_label = torch.cat(out_labellist).cpu()
    class_label = torch.cat(labellist).cpu()
    class_ID = torch.cat(IDlist).cpu()
    class_preds = torch.cat(predlist).cpu()
    class_probs = torch.cat([torch.stack(batch) for batch in problist])

    loss_cls_vaule = loss_cls(class_out_label,class_label)


    test_ID_label = {}
    test_ID_preds = {}
    csvf = open('./trails/models/**/testpreds_{}epoch.csv'.format(epoch),'w')
    fileheader = ['ID','pred']
    dict_writer = csv.DictWriter(csvf,fileheader)
    dict_writer.writerow(dict(zip(fileheader, fileheader)))
    # print(set(test_ID.numpy().tolist()))
    for i_d in set(class_ID.numpy().tolist()):
        idx = (class_ID == i_d).nonzero(as_tuple=False)
        # test_ID_label[i_d]= class_label[idx].squeeze()
        test_ID_preds[i_d]= class_preds[idx].squeeze()
        dict_writer.writerow({"ID": int(i_d), "pred": test_ID_preds[i_d]})
    # print(test_ID_label)
    # print(test_ID_preds)
    csvf.close()

    acc_score = accuracy_score(class_label, class_preds)
    cm = confusion_matrix(class_label, class_preds, labels=None, sample_weight=None)
    print(cm)
    # precision = precision_score(class_label, class_preds, average='weighted')
    # recall = recall_score(class_label, class_preds, average='weighted')
    # print(acc_score)

    # ??????ROC AUC  ROC ??????????????????1????????????????????????????????????
    class_label_auc = class_label.detach().numpy()
    class_probs_auc = class_probs[:,1].cpu().detach().numpy()
    fpr, tpr, thersholds = roc_curve(class_label_auc,class_probs_auc)
    auc_score = auc(fpr, tpr)
    pre, rec, _ = precision_recall_curve(class_label_auc,class_probs_auc)
    AP = average_precision_score(class_label_auc,class_probs_auc)

    log.info('loss_seg:{:.3f}\tDSC_sum:{:.4f}\tPPV_sum:{:.4f}\tSen_sum:{:.4f}\tloss_cls:{:.3f}\tacc:{:.4f}\tAUC:{:.4f}\tAP:{:.4f}'
            .format(loss_seg_sum,DSC_sum, PPV_sum, Sen_sum,loss_cls_vaule,acc_score,auc_score,AP))
    return acc_score,auc_score,loss_cls_vaule


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 
    model = model.cuda() 
    model = nn.DataParallel(model)
    # optimizer
    if sets.pretrain_path:
        params = [
                    { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
                    { 'params': parameters['new_parameters'], 'lr': sets.learning_rate}
                    ]
    else:
        params = [{'params': parameters, 'lr': sets.learning_rate}]

    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True

    # testing
    sets.phase = 'test'
    testing_dataset = CPDataset(sets.data_root, sets.img_list_test, sets)
    test_loader = DataLoader(testing_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    model, optimizer = resume('./trails/models/trained_parm*.tar', model ,optimizer)

    acc_score,auc_score,loss = test(test_loader,model,sets,3)
