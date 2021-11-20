import torch
from torch import nn
from models.resnet3D import resnet10,resnet18,resnet34,resnet50,resnet101,resnet200,resnet152
# Twopath_only
# from models.twopath_only import TwoPath
# Twopath_Unet_coord
# from models.twopath_Unet_coord import TwoPath
# # Twopath_Unet_coord_mask
from models.twopath_Unet_coord_mask import TwoPath

# only resnet2D
from models.resnet2D_only import MyResModel2D

# Unet
# from models.resnet2D_unet_only_4 import MyResModel2DUnetOnly
# Unet+Coordconv
# from models.unet_only_coord_4 import MyResModel2DUnetOnly



def generate_model(opt):
    assert opt.model in [
        'resnet','twopath','resnet2D','Unet','Unet_only'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
    elif opt.model == 'twopath':
        model = TwoPath(opt)
    elif opt.model == 'resnet2D':
        model = MyResModel2D(opt)
    elif opt.model == 'Unet':
        model = MyResModel2DUnetOnly(opt)
    elif opt.model == 'Unet_only':
        model = MyResModel2DUnetOnly(opt)
    
    if opt.model =="twopath":
        new_layer_names = ['myRes2D.0.myresnet.fc']
        new_parameters = [] 
        for pname, p in model.named_parameters():
            # print(pname,p.shape,p.requires_grad)
            for layer_name in new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        # print(new_parameters_id)
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()
