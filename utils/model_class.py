import sys, os
import numpy as np
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.convnext import LayerNorm2d
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.models import swin_s, Swin_S_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from . import bit_models


def get_arch(model_name, n_classes=1, pretrained=False):
    weights = None
    if model_name == 'resnet18':
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'resnet34':
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
        model = resnet34(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'resnet50':
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'mobilenet_v2':
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    elif model_name == 'eff_b1':
        if pretrained:
            weights = EfficientNet_B1_Weights.IMAGENET1K_V2
        model = efficientnet_b1(weights=weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    elif model_name == 'eff_b2':
        if pretrained:
            weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        model = efficientnet_b2(weights=weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    elif model_name == 'eff_v2':
        if pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=n_classes)


    elif model_name == 'eff_v2_M':
        if pretrained:
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        
        model = efficientnet_v2_m(weights=weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    elif model_name == 'convnext':
        if pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        
        model = convnext_tiny(weights=weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Sequential(LayerNorm2d([num_ftrs, ], eps=1e-06, elementwise_affine=True),
                                         nn.Flatten(start_dim=1, end_dim=-1),
                                         nn.Linear(in_features=768, out_features=n_classes))

    elif model_name == 'swin':
        if pretrained:
            weights = Swin_T_Weights.IMAGENET1K_V1
        
        model = swin_t(weights=weights)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    elif model_name == 'swinV2':
        if pretrained:
            weights = Swin_V2_T_Weights.IMAGENET1K_V1

        model = swin_v2_t(weights=weights)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    elif model_name == 'swinS':
        if pretrained:
            weights = Swin_S_Weights.IMAGENET1K_V1
        
        model = swin_s(weights=weights)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    elif model_name == 'bit_rx50':
        model = bit_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=n_classes, zero_head=True)
        if pretrained:
            if not os.path.isfile('models/BiT-M-R50x1.npz'):
                print('downloading bit_resnext50_1 weights:')
                os.system('wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz -P models/')
            model.load_from(np.load('models/BiT-M-R50x1.npz'))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    else:
        sys.exit('{} is not a valid model_name, check utils.get_model_v2.py'.format(model_name))

    setattr(model, 'n_classes', n_classes)
    return model



