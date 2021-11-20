from setting import parse_opts 
from datasets.CP_npy import CPDataset 
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

    weights = [1.0,1.54]
    class_weights = torch.FloatTensor(weights).cpu()
    loss_cls = nn.CrossEntropyLoss(weight=class_weights,ignore_index=-1)

    for i,data in enumerate(testloader):
        volumes_all,mask_all,label,ID = data
        if not sets.no_cuda: 
            volumes_all = volumes_all.cuda()
        
        with torch.no_grad():
            if sets.model =="twopath":
                out_label = model(volumes_all,volumes_all)
            else:
                out_label = model(volumes_all)

            prob =[F.softmax(el,dim=0) for el in out_label]

            _, preds = torch.max(out_label, 1)
            out_labellist.append(out_label)
            predlist.append(preds)
            labellist.append(label)
            IDlist.append(ID)
            problist.append(prob)

    class_out_label = torch.cat(out_labellist).cpu()
    class_label = torch.cat(labellist).cpu()
    class_ID = torch.cat(IDlist).cpu()



    class_preds = torch.cat(predlist).cpu()
    class_probs = torch.cat([torch.stack(batch) for batch in problist])

    loss = loss_cls(class_out_label,class_label)


    test_ID_label = {}
    test_ID_preds = {}
    test_ID_prob = {}
    csvf = open('./trails/models/**/testpreds_{}epoch.csv'.format(epoch),'w')
    fileheader = ['ID','pred','probility']
    dict_writer = csv.DictWriter(csvf,fileheader)
    dict_writer.writerow(dict(zip(fileheader, fileheader)))
    # print(set(test_ID.numpy().tolist()))
    for i_d in set(class_ID.numpy().tolist()):
        idx = (class_ID == i_d).nonzero(as_tuple=False)
        # test_ID_label[i_d]= class_label[idx].squeeze()
        test_ID_preds[i_d]= class_preds[idx].squeeze()
        test_ID_prob[i_d] = class_probs[idx].squeeze()
        dict_writer.writerow({"ID": int(i_d), "pred": test_ID_preds[i_d],'probility':test_ID_prob[i_d]})
    # print(test_ID_label)
    # print(test_ID_preds)
    csvf.close()

    acc_score = accuracy_score(class_label, class_preds)
    cm = confusion_matrix(class_label, class_preds, labels=None, sample_weight=None)
    print(cm)
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


    log.info('loss:{:.4f}\tacc:{:.4f}\tAUC:{:.4f}\tAP:{:.4f}'.format(loss,acc_score,auc_score,AP))
    return acc_score,auc_score,loss


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    tmp = ''
    for id in sets.gpu_id:
        tmp += str(id)+','
   
    os.environ["CUDA_VISIBLE_DEVICES"]= tmp
    sets.target_type = "normal"
    sets.phase = 'test'
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 
    model = model.cuda() 
    model = nn.DataParallel(model)
    # optimizer
    if sets.model =="twopath":
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

    model, optimizer = resume('./trails/models/**/trained_parm*.tar', model ,optimizer)
    acc_score,auc_score,loss = test(test_loader,model,sets,3)
