# All these are from here:
# https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/330336

import torch
import torch.nn.functional as F


# def partial_bce_with_logits(inputs, targets, reduction='mean'):
#     # pick predictions on positive targets (k) - with an eye on label smoothing
#
#     inputs_on_pos_targets = inputs[(targets > 0.).long()]
#     inputs_on_neg_targets = inputs[(targets == 0.).long()]
#     # pick top-k predictions on negative targets (worse k predictions)
#     topk_inputs_on_neg_targets = torch.topk(inputs_on_neg_targets,
#                                             k=min(len(inputs_on_neg_targets), targets.count_nonzero())).values
#
#     targets = torch.cat([torch.ones_like(inputs_on_pos_targets), torch.zeros_like(topk_inputs_on_neg_targets)])
#     inputs = torch.cat([inputs_on_pos_targets, topk_inputs_on_neg_targets])
#
#     return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)

def partial_bce_with_logits(inputs, targets, reduction='mean'):
    # memory issues if type is monai.data.meta_tensor.MetaTensor, apparently
    # hey if patch only has zeros we do not learn shit?
    inputs, targets = torch.Tensor(inputs), torch.Tensor(targets)
    n_nonzeros = targets.count_nonzero()
    if n_nonzeros == 0:
        return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)

    n_zeros = targets.numel() - n_nonzeros
    k = min(n_zeros, n_nonzeros)

    return F.binary_cross_entropy_with_logits(
        torch.cat([torch.topk(inputs[targets == 0], k).values, inputs[targets > 0]]).to(inputs.device),
        torch.cat([torch.zeros(k), torch.ones(n_nonzeros)]).to(inputs.device).float(), reduction=reduction)

def bce(inputs, targets):
    return -torch.mean(targets * torch.log(inputs) + (1. - targets) * torch.log(1. - inputs))

def jacc(inputs, targets, smooth = 1):
    inputs = F.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

def dice(inputs, targets, smooth=1e-6):
    inputs = F.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice

def focal_loss(_in, target, alpha = 0.5, gamma = 2):
    logprobs = torch.nn.functional.binary_cross_entropy_with_logits(_in, target)
    return torch.mean(torch.pow((1 - torch.sigmoid(logprobs.sum(dim = 1))), gamma) * logprobs)

def tversky(inputs, targets, alpha = 0.5, beta = 0.5, smooth=1e-6):
    inputs = F.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    denominator = (inputs * targets).sum() + alpha * (1 - inputs) * targets.sum() + beta * inputs * (1 - targets).sum()
    return 1 - (intersection + smooth) / (denominator + smooth)

def lovasz_hinge(inputs, targets):
    inputs = F.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    loss = (1 - targets) * (inputs - 1) + (1 + targets) * F.relu(inputs)
    return loss.mean()
