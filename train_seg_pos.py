"""
@author: Modified by A. Galdran starting from the code provided by SHIFTS:
https://github.com/Shifts-Project/shifts/tree/main/mswml
"""

import torch
from torch import nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
import json
import os, os.path as osp
import sys
import numpy as np
import random
# from utils.metrics import dice_metric, dice_norm_metric, fast_dice_metric, fast_auc
from utils.data_load import get_train_val_dataloaders
from utils.model import get_model
from utils.sam import SAM
import os, os.path as osp, sys, json, time
from tqdm import trange
import warnings
from utils.data_load import get_train_val_pos_dataloaders
# from utils.metrics import dice_norm_metric, lesion_f1_score, ndsc_aac_metric
from joblib import Parallel
from skimage.filters import threshold_otsu
from sklearn.metrics import log_loss
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from datetime import datetime
import shutil
from monai.transforms import RandGaussianSmooth, GaussianSmooth
from monai.networks.layers import GaussianFilter
from typing import Sequence
from monai.networks.layers import GaussianFilter
from torch.optim.swa_utils import AveragedModel

from utils.val_script import false_neg_pix, false_pos_pix
from scipy.stats import rankdata


def fast_auc(actual, predicted, partial=False):
    actual, predicted = actual.flatten(), predicted.flatten()
    if partial:
        n_nonzeros = np.count_nonzero(actual)
        n_zeros = len(actual) - n_nonzeros
        k = min(n_zeros, n_nonzeros)
        predicted = np.concatenate([
            np.sort(predicted[actual == 0])[::-1][:k],
            np.sort(predicted[actual == 1])[::-1][:k]
        ])
        actual = np.concatenate([np.zeros(k), np.ones(k)])

    r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    if n_pos == 0 or n_neg == 0: return 0
    return (np.sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def fast_dice(actual, predicted):
    actual = np.asarray(actual).astype(bool)
    predicted = np.asarray(predicted).astype(bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0: return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum

def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum

    return dice_score
def get_challenge_metrics(actual, predicted, voxel_vol):

    false_neg_vol = false_neg_pix(actual, predicted) * voxel_vol
    false_pos_vol = false_pos_pix(actual, predicted) * voxel_vol

    return false_neg_vol, false_pos_vol

def get_args_parser():
    import argparse
    def str2bool(v):
        # as seen here: https://stackoverflow.com/a/43357954/3208255
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', 'yes'):
            return True
        elif v.lower() in ('false', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('boolean value expected.')

    parser = argparse.ArgumentParser(description='Multiple Sclerosis Lesion Segmentation')
    # data
    parser.add_argument('--fold', type=int, default=1, help='Specify the train/val split')
    parser.add_argument('--path_csv_train', type=str, default='data/', help='Path to csv')
    parser.add_argument('--proportion', default=1.0, type=float, help='how much of data to use 0 to 1 (default: 100%)')
    parser.add_argument('--n_samples', type=int, default=12, help='nr of patches extracted per loaded volume before moving to the next one')
    parser.add_argument('--neg_samples', type=int, default=2, help='when sampling, P(center=fground) = 1/(1+neg_samples)')
    parser.add_argument('--tr_batch_size', type=int, default=4, help='patches are fed to model in groups of batch_size')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers (default: 10)')
    parser.add_argument('--cache', default=0.0, type=float, help='data to be cached (default: 0%)')
    parser.add_argument('--persist', type=str2bool, nargs='?', const=True, default=False, help='persistent cache')
    parser.add_argument('--notest', type=str2bool, nargs='?', const=True, default=False, help='persistent cache')

    # initialization
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
    parser.add_argument('--model', type=str, default='swinunetr_tiny', help='architecture')
    parser.add_argument('--patch_size', type=str, default='0/0/0', help='volumetric patch sampling, 0/0/0 picks per-model defaults')
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='use pretrained weights if available')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='path to checkpoint for resuming training')

    parser.add_argument('--n_classes', type=int, default=1, help='1 uses sigmoid, 2 uses softmax')
    parser.add_argument('--n_heads', type=int, default=1, help='how many heads in model exit')
    parser.add_argument('--cycle_lens', type=str, default='5/1', help='cycling config (nr cycles/cycle len)')

    # training
    parser.add_argument('--loss', default='ce', type=str, choices=('ce', 'pbce', 'dice', 'dice_ce', 'dice_focal', 'focal'), help='loss function')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha in a compound loss')  # adam -> 1e-4, 3e-4
    parser.add_argument('--beta',  type=float, default=1.0, help='beta in a compound loss')  # adam -> 1e-4, 3e-4

    parser.add_argument('--opt', default='nadam', type=str, choices=('sgd', 'adamw', 'nadam'), help='optimizer to use (sgd | adamW)')
    parser.add_argument('--sam', type=str2bool, nargs='?', const=True, default=False, help='use sam wrapping optimizer')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')  # adam -> 1e-4, 3e-4
    parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

    # logging
    parser.add_argument('--save_path', type=str, default='last_experiment')
    parser.add_argument('--fast_eval', type=str2bool, nargs='?', const=True, default=True, help='use fast evaluation (dice and nll)')


    args = parser.parse_args()

    return args

def set_seed(seed_value):
    if seed_value is not None:
        np.random.seed(seed_value)  # cpu vars
        torch.manual_seed(seed_value)  # cpu  vars
        random.seed(seed_value)  # Python
        # torch.use_deterministic_algorithms(True)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def disable_bn(model):
    for module in model.modules():
      if isinstance(module, torch.nn.BatchNorm3d):
        module.eval()
def enable_bn(model):
    model.train()

# from utils.evaluation import get_tr_info
def init_tr_info():
    tr_info = dict()
    tr_info['tr_patch_losses'], tr_info['tr_patch_dices'] = [], []
    tr_info['tr_aucs'], tr_info['tr_pos_dices'] = [], []
    tr_info['vl_aucs'], tr_info['vl_pos_dices'] = [], []

    return tr_info

def set_tr_info(tr_info, vl_pos_metrics=None, ovft_metrics=None, tr_patch_loss=0., tr_patch_dice=0., best_cycle=False):

    if best_cycle:
        tr_info['best_patch_loss'] = tr_info['tr_patch_losses'][-1]
        tr_info['best_patch_dice'] = tr_info['tr_patch_dices'][-1]
        tr_info['best_auc'] = tr_info['vl_aucs'][-1]
        tr_info['best_pos_dice'] = tr_info['vl_pos_dices'][-1]
        tr_info['best_tr_auc'] = tr_info['tr_aucs'][-1]
        tr_info['best_tr_pos_dice'] = tr_info['tr_pos_dices'][-1]
        tr_info['best_cycle'] = len(tr_info['vl_aucs'])

    else:
        tr_info['tr_patch_losses'].append(tr_patch_loss)
        tr_info['tr_patch_dices'].append(tr_patch_dice)
        tr_info['vl_aucs'].append(vl_pos_metrics[0])
        tr_info['vl_pos_dices'].append(vl_pos_metrics[1])
        tr_info['tr_aucs'].append(ovft_metrics[0])
        tr_info['tr_pos_dices'].append(ovft_metrics[1])

    return tr_info

def get_eval_string(tr_info, cycle):
    # pretty prints first three values of train/val metrics to a string and returns it
    s = 'Cycle {}: Tr/Vl AUC: {:.2f}/{:.2f} - DSC: {:.2f}/{:.2f} - Tr Patch-DSC: {:.2f} - Tr Patch-Loss: {:.4f}'.format(
        str(cycle+1).zfill(2), tr_info['tr_aucs'][cycle], tr_info['vl_aucs'][cycle], tr_info['tr_pos_dices'][cycle],
              tr_info['vl_pos_dices'][cycle], tr_info['tr_patch_dices'][cycle], tr_info['tr_patch_losses'][cycle])
    return s



def validate_pos(model, loader, slwin_batch_size, fast_eval=True):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    thresh = 0.25
    patch_size = model.patch_size
    aucs, dices = [], []
    best_dices, best_ts, otsu_ts, otsu_dices, otsu_better = [], [], [], [], []
    with trange(len(loader)) as t:
        n_elems, running_dice = 0, 0
        for val_data in loader:
            # pixdim = val_data['ct_suv'].pixdim[0]
            # voxel_vol = torch.prod(pixdim).item()/1000.
            val_input, gt_array = val_data['ct_suv'].to(device), (val_data['seg'].squeeze().numpy() > thresh).astype(np.uint8)
            prob_array = sliding_window_inference(val_input.to(device), patch_size, slwin_batch_size, model,
                                                  mode='gaussian')
            prob_array = prob_array.squeeze().sigmoid().cpu().numpy()
            if gt_array.sum() == 0: print('wtf pos')
            aucs.append(fast_auc(gt_array, prob_array, partial=True))
            pred_array = (prob_array > thresh).astype(np.uint8)
            d_score = fast_dice(gt_array, pred_array)
            dices.append(d_score)

            d_best, t_best = 0, 0
            for th in np.arange(0.,0.51, 0.05):
                seg = (prob_array > th).astype(np.uint8)
                d = fast_dice(gt_array, seg)
                if d > d_best:
                    d_best, t_best = d, th
            best_dices.append(d_best)
            best_ts.append(t_best)

            t_otsu = threshold_otsu(prob_array)
            seg = (prob_array > t_otsu).astype(np.uint8)
            d_otsu = fast_dice(gt_array, seg)
            otsu_ts.append(t_otsu)
            otsu_dices.append(d_otsu)

            if d_otsu > d_score:
                otsu_better.append(1)
            else:
                otsu_better.append(0)
            n_elems += 1
            running_dice += d_score
            run_dice = running_dice / n_elems
            t.set_postfix(DSC="{:.4f}".format(100 * run_dice))
            t.update()

    m = np.array(list(zip(best_ts, best_dices, otsu_ts, otsu_dices, otsu_better)))

    return [100 * np.mean(np.array(aucs)), 100 * np.mean(np.array(dices))], m


def train_one_epoch(model, tr_loader, tr_batch_size, loss_fn, optimizer, scheduler):
    # monai complains when no foreground and asked for sampling from foreground
    warnings.filterwarnings("ignore", category=UserWarning)
    model.train()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    dice_metric = DiceMetric(ignore_empty=False)
    thresh = 0.25
    with trange(len(tr_loader)) as t:
        step, n_elems, running_loss, running_dice = 0, 0, 0, 0
        for batch_data in tr_loader:  # load 1 scan from the training set
            n_samples = len(batch_data['seg'])  # nr of px x py x pz patches (see args.n_samples)
            for m in range(0, n_samples, tr_batch_size):  # we loop over batch_data picking up tr_batch_size patches at a time
                step += tr_batch_size
                inputs, labels = (batch_data['ct_suv'][m:(m+tr_batch_size)].to(device), batch_data['seg'][m:(m+tr_batch_size)].to(device))

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                if isinstance(optimizer, SAM):
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                    disable_bn(model)
                    loss.backward()
                    enable_bn(model)
                    optimizer.second_step(zero_grad=True)
                else:
                    loss.backward()
                    optimizer.step()
                lr = get_lr(optimizer)
                scheduler.step()
                optimizer.zero_grad()


                seg = outputs.sigmoid()  # P(foreground))
                seg = (seg > thresh).long()
                labels = (labels >thresh).long()  # this is for when we use label smoothing
                dice_metric(y_pred=seg, y=labels)
                dice_score = dice_metric.aggregate().item()
                if np.isnan(dice_score): dice_score = 0

                running_loss += loss.detach().item() * inputs.shape[0]
                running_dice += dice_score
                n_elems += inputs.shape[0]  # total nr of items processed
                run_loss = running_loss / n_elems
                run_dice = running_dice / n_elems


            t.set_postfix(LOSS_DICE_lr="{:.4f}/{:.4f}/{:.6f}".format(run_loss, 100*run_dice, lr))
            t.update()

        return run_loss, 100*run_dice

def train_one_cycle(model, tr_loader, tr_batch_size, loss_fn, optimizer, scheduler, cycle):
    model.train()
    cycle_len = scheduler.cycle_lens[cycle]
    for epoch in range(cycle_len):
        print('Cycle {:d} | Epoch {:d}/{:d}'.format(cycle+1, epoch+1, cycle_len))
        tr_patch_loss, tr_patch_dice = train_one_epoch(model, tr_loader, tr_batch_size, loss_fn, optimizer, scheduler)
    return tr_patch_loss, tr_patch_dice

def train(model, tr_loader, vl_pos_loader, ovft_dl, tr_batch_size, slwin_batch_size, loss_fn, optimizer, scheduler, fast_eval, save_path):

    best_metric, best_cycle = 0, 0
    n_cycles = len(scheduler.cycle_lens)
    tr_info = init_tr_info()
    for cycle in range(n_cycles):
        print('\nCycle {:d}/{:d}'.format(cycle+1, n_cycles))
        # train one cycle
        tr_patch_loss, tr_patch_dice = train_one_cycle(model, tr_loader, tr_batch_size, loss_fn, optimizer, scheduler, cycle)

        with torch.inference_mode():
            ovft_metrics, ovft_dices = validate_pos(model, ovft_dl, slwin_batch_size, fast_eval)
            vl_pos_metrics, vl_dices = validate_pos(model, vl_pos_loader, slwin_batch_size, fast_eval)

        tr_info = set_tr_info(tr_info, vl_pos_metrics, ovft_metrics, tr_patch_loss=tr_patch_loss, tr_patch_dice=tr_patch_dice)

        s = get_eval_string(tr_info, cycle)
        print(s)

        ### log dice scores with different thresholding strategies ###
        # np.save(os.path.join(save_path, 'ovft_dices_c{}.npy'.format(cycle + 1)), ovft_dices)
        # np.save(os.path.join(save_path, 'vl_dices_c{}.npy'.format(cycle+1)), vl_dices)

        m_ovft = np.median(ovft_dices, axis=0)
        m_vl = np.median(vl_dices, axis=0)
        s2 = ' ** Best Ovft manual Th/DSC = {:.2f}/{:.2f} || Avg Otsu Th/DSC = {:.2f}/{:.2f} **  '.format(m_ovft[0], 100*m_ovft[1], m_ovft[2], 100*m_ovft[3])
        s3 = ' ** Best Val. manual Th/DSC = {:.2f}/{:.2f} || Avg Otsu Th/DSC = {:.2f}/{:.2f} **  '.format(m_vl[0], 100*m_vl[1], m_vl[2], 100*m_vl[3])
        print(s2)
        print(s3)
        ###

        with open(osp.join(save_path, 'train_log.txt'), 'a') as f:
            print(s, file=f)
            print(s2, file=f)
            print(s3, file=f)

        if vl_pos_metrics[0] > best_metric:
            print('-------- Best metric attained. {:.2f} --> {:.2f} --------'.format(best_metric, vl_pos_metrics[0]))
            best_metric = vl_pos_metrics[0]
            best_cycle = cycle+1
            tr_info = set_tr_info(tr_info, vl_pos_metrics, cycle, best_cycle=True)
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            print('-------- Best metric so far {:.2f} at cycle {:d} --------'.format(best_metric, best_cycle))

    torch.save(model.state_dict(), os.path.join(save_path, "last_model.pth"))
    return tr_info



def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # reproducibility
    set_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system') # not sure what this does

    #
    print('GPU: {}'.format(torch.cuda.get_device_name(0)))
    # Use all cores
    print('CPU Count: {}'.format(os.cpu_count()))
    torch.set_num_threads(os.cpu_count())
    print('Num threads: {}'.format(torch.get_num_threads()))

    save_path = osp.join('experiments', args.save_path)
    os.makedirs(save_path, exist_ok=True)

    ''' model '''
    patch_size = args.patch_size.split('/')
    patch_size = tuple(map(int, patch_size))

    model = get_model(args.model, patch_size, args.pretrained, args.n_classes, n_heads=args.n_heads)
    model.to(device)
    # print(model)
    if args.load_checkpoint is not None:
        try: model.load_state_dict(torch.load(osp.join(args.load_checkpoint, 'last_model.pth')))
        except: sys.exit('Specified checkpoint does not work with this model, please check it.')

    '''' dataloaders '''
    n_samples, neg_samples, tr_batch_size, patch_size, slwin_batch_size = args.n_samples, args.neg_samples, args.tr_batch_size, model.patch_size, 4

    tr_loader, vl_loader, ovft_dl = get_train_val_pos_dataloaders(fold=args.fold, n_samples=n_samples, neg_samples=neg_samples,
                                                        patch_size=patch_size, proportion=args.proportion, notest=args.notest,
                                                        num_workers=args.num_workers, cache=args.cache, persist=args.persist)
    ''' logging '''
    config_file_path = osp.join(save_path, 'config.cfg')
    with open(config_file_path, 'w') as f:
        json.dump(vars(args), f, indent=2)



    if args.loss == 'ce':
        loss_fn = torch.nn.BCEWithLogitsLoss() if model.n_classes == 1 else torch.nn.CrossEntropyLoss()
        # loss_fn = DiceCELoss(sigmoid=True, lambda_dice=0.0, lambda_ce=1.0)
    elif args.loss == 'pbce':
        from utils.losses import partial_bce_with_logits
        loss_fn = partial_bce_with_logits
    elif args.loss == 'dice':
        loss_fn = DiceLoss(sigmoid=True)
    elif args.loss == 'dice_ce':
        loss_fn = DiceCELoss(sigmoid=True, lambda_dice=args.alpha, lambda_ce=args.beta)
        # note that DiceCELoss does compute CE on the background pixels too (!!)
    elif args.loss == 'dice_focal':
        loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=args.alpha, lambda_focal=args.beta)
    elif args.loss == 'focal':
        loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=0.0, lambda_focal=1.0)
    else:
        sys.exit('invalid loss choice')
    ''' optimizer '''
    # Prepare optimizer and scheduler
    if args.opt == 'adamw':
        if args.sam:
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'nadam':
        if args.sam:
            base_optimizer = torch.optim.NAdam
            optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        if args.sam:
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise RuntimeError('Invalid optimizer {}. Only SGD and AdamW are supported.'.format(args.opt))

    cycle_lens = args.cycle_lens.split('/')
    cycle_lens = list(map(int, cycle_lens))
    if len(cycle_lens) > 2:
        sys.exit('cycles should be specified as a pair n_cycles/cycle_len')
    cycle_lens = cycle_lens[0] * [cycle_lens[1]]
    if isinstance(optimizer, SAM):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=cycle_lens[0]*len(tr_loader)*args.n_samples//tr_batch_size, eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle_lens[0]*len(tr_loader)*args.n_samples//tr_batch_size, eta_min=0)

    setattr(optimizer, 'max_lr', args.lr)  # store maximum lr inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)


    # Start training
    start = time.time()
    tr_info = train(model, tr_loader, vl_loader, ovft_dl, tr_batch_size, slwin_batch_size, loss_fn, optimizer, scheduler, args.fast_eval, save_path)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)


    with (open(osp.join(save_path, 'log.txt'), 'w') as f):
        print('Best cycle = {}/{} -- Best Tr/Vl AUC = {:.2f}/{:.2f} -- Tr/Vl DSC = {:.2f}/{:.2f}\n'\
                'Best Tr Patch DSC = {:.2f}\nTr Patch loss = {:.6f} \n'.format(tr_info['best_cycle'], len(cycle_lens),
                tr_info['best_tr_auc'], tr_info['best_auc'], tr_info['best_tr_pos_dice'], tr_info['best_pos_dice'],
                tr_info['best_patch_dice'], tr_info['best_patch_loss']), file=f)

        tr_aucs, vl_aucs, tr_pos_dices, vl_pos_dices, tr_patch_losses, tr_patch_dices = tr_info['tr_aucs'], tr_info['vl_aucs'],  \
                    tr_info['tr_pos_dices'], tr_info['vl_pos_dices'], tr_info['tr_patch_losses'], tr_info['tr_patch_dices']
        for j in range(len(vl_aucs)):
            print('Cycle = {} -> Tr/Vl AUC={:.2f}/{:.2f}, DSC={:.2f}/{:.2f}, Tr Patch DSC={:.2f}, Tr Patch Loss={:.4f}'.format(j+1,
                   tr_aucs[j], vl_aucs[j], tr_pos_dices[j], vl_pos_dices[j], tr_patch_dices[j], tr_patch_losses[j]), file=f)
        print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)

    print('Done. Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))

if __name__ == "__main__":
    args = get_args_parser()
    main(args)

