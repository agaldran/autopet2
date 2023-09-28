import os, json, sys, time, random, os.path as osp
import numpy as np
import torch
from tqdm import trange, tqdm
# from utils.data_handling import get_class_loaders
# from utils.evaluation import evaluate_cls
from utils.model_class import get_arch
from utils.data_load import get_train_val_class_dataloaders
from utils.sam import SAM
from scipy.stats import pearsonr, spearmanr, kendalltau

def set_seed(seed_value, use_cuda):
    if seed_value is not None:
        np.random.seed(seed_value)  # cpu vars
        torch.manual_seed(seed_value)  # cpu  vars
        random.seed(seed_value)  # Python
        # torch.use_deterministic_algorithms(True)

        if use_cuda:
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True

def save_model(path, model):
    os.makedirs(path, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()},
                 osp.join(path, 'model_checkpoint.pth'))

def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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

    parser = argparse.ArgumentParser(description='Training for Biomedical Image Classification')
    parser.add_argument('--fold', type=int, default=1, help='Specify the train/val split')
    parser.add_argument('--path_csv_train', type=str, default='data/', help='Path to csv')
    parser.add_argument('--projection', help='which mip projection to use on PET data', type=str, choices=('x', 'y', 'z'), default='x')
    parser.add_argument('--proportion', default=1.0, type=float, help='how much of data to use 0 to 1 (default: 100%)')

    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers (default: 10)')
    parser.add_argument('--cache', default=0.0, type=float, help='data to be cached (default: 0%)')
    parser.add_argument('--persist', type=str2bool, nargs='?', const=True, default=False, help='persistent cache')

    # initialization
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
    parser.add_argument('--model', type=str, default='mobilenet_v2', help='architecture')
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='use imagenet weights')
    parser.add_argument('--n_classes', type=int, default=1, help='1 uses sigmoid, 2 uses softmax')
    parser.add_argument('--balanced_weights', type=str2bool, nargs='?', const=True, default=False, help='noisy heads (default), or balanced')
    parser.add_argument('--cycle_lens', type=str, default='2/1', help='cycling config (nr cycles/cycle len)')

    parser.add_argument('--loss', type=str, default='ce', help='overall loss on top of head losses')
    parser.add_argument('--hypar', type=float, default=-1, help='some overall losses have hyper-parameter, set -1 for their defaults')
    parser.add_argument('--opt', default='nadam', type=str, choices=('sgd', 'adamw', 'nadam'), help='optimizer to use (sgd | adamW | nadam)')
    parser.add_argument('--sam', type=str2bool, nargs='?', const=True, default=False, help='use sam wrapping optimizer')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')  # adam -> 1e-4, 3e-4
    parser.add_argument('--momentum', default=0., type=float, help='sgd momentum')
    parser.add_argument('--epsilon', default=1e-8, type=float, help='adamW epsilon for numerical stability')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='label smoothing (default: 0.0)')

    parser.add_argument('--save_path', type=str, default=None, help='path to save model (defaults to None => debug mode)')
    parser.add_argument('--device', default='cuda', type=str, help='device (cuda or cpu, default: cuda)')

    args = parser.parse_args()

    return args

def disable_bn(model):
    for module in model.modules():
      if isinstance(module, torch.nn.BatchNorm3d):
        module.eval()
def enable_bn(model):
    model.train()

def run_one_epoch(model, optimizer, criterion, loader, scheduler=None, assess=False):

    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train: model.train()
    else: model.eval()

    probs_class_all, preds_class_all, labels_all = [], [], []
    run_loss_class = 0
    with trange(len(loader)) as t:
        n_elems, running_loss_class = 0, 0
        for i_batch, batch_data in enumerate(loader):
            inputs, labels = batch_data['suv'].to(device), batch_data['positive'].float().to(device)
            logits = model(inputs)

            loss_class = criterion(logits.squeeze(), labels.squeeze())

            if train:  # only in training mode
                loss_class.backward()
                if isinstance(optimizer, SAM):
                    optimizer.first_step(zero_grad=True)
                    logits = model(inputs)
                    loss_class = criterion(logits.squeeze(), labels.squeeze())
                    # compute BN statistics only in the first backwards pass
                    disable_bn(model)
                    loss_class.backward()
                    enable_bn(model)
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()
                lr = get_lr(optimizer)
                scheduler.step()
                optimizer.zero_grad()
            if assess:
                probs_class = logits.sigmoid().squeeze().detach().cpu().numpy()
                preds_class = (probs_class>0.5).astype(int)
                labels = labels.squeeze().cpu().numpy()

                if labels.size == 1:
                    probs_class_all.append(probs_class.item())
                    preds_class_all.append(preds_class.item())
                    labels_all.append(labels.item())
                else:
                    probs_class_all.extend(probs_class)
                    preds_class_all.extend(preds_class)
                    labels_all.extend(labels)


            # Compute running loss
            running_loss_class += loss_class.detach().item() * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss_class = running_loss_class / n_elems
            if train: t.set_postfix(loss_lr="{:.4f}/{:.6f}".format(run_loss_class, lr))
            else: t.set_postfix(vl_loss="{:.4f}".format(float(run_loss_class)))
            t.update()
    if assess:
        try:
            a=np.stack(probs_class_all)
        except:
            print(probs_class_all)
            sys.exit()
    if assess: return np.stack(probs_class_all).flatten(), np.stack(preds_class_all).flatten(), np.stack(labels_all).flatten(), run_loss_class
    return None, None, None, None

def train_one_cycle(model, optimizer, criterion, train_loader, scheduler, cycle=0):

    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]

    for epoch in range(cycle_len):
        print('Cycle {:d} | Epoch {:d}/{:d}'.format(cycle+1, epoch+1, cycle_len))
        if epoch == cycle_len-1: assess=True # only get probs/preds/labels on last cycle
        else: assess = False
        tr_probs, tr_preds, tr_labels, tr_loss = run_one_epoch(model, optimizer, criterion, train_loader, scheduler, assess)

    return tr_preds, tr_probs, tr_labels, tr_loss

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
def get_class_metrics(probs, preds, labels):
    try:
        auc = roc_auc_score(labels, probs)
    except:
        print(np.unique(labels, return_counts=True))
        auc = 0
    try:
        bacc = (labels, preds)
    except:
        print(labels.shape, preds.shape)
        sys.exit()
    mcc = matthews_corrcoef(labels, preds)
    return 100*auc, 100*bacc, 100*mcc

def init_tr_info():
    tr_info = dict()
    tr_info['tr_losses'], tr_info['vl_losses'] = [], []
    tr_info['tr_aucs'], tr_info['tr_baccs'], tr_info['tr_mccs'] = [], [], []
    tr_info['vl_aucs'], tr_info['vl_baccs'], tr_info['vl_mccs'] = [], [], []

    return tr_info

def set_tr_info(tr_info, vl_metrics=None, ovft_metrics=None, tr_loss=0., vl_loss=0., best_cycle=False):

    if best_cycle:
        tr_info['best_tr_loss'] = tr_info['tr_losses'][-1]
        tr_info['best_tr_auc'] = tr_info['tr_aucs'][-1]
        tr_info['best_tr_bacc'] = tr_info['tr_baccs'][-1]
        tr_info['best_tr_mcc'] = tr_info['tr_mccs'][-1]

        tr_info['best_vl_loss'] = tr_info['vl_losses'][-1]
        tr_info['best_vl_auc'] = tr_info['vl_aucs'][-1]
        tr_info['best_vl_bacc'] = tr_info['vl_baccs'][-1]
        tr_info['best_vl_mcc'] = tr_info['vl_mccs'][-1]
        tr_info['best_cycle'] = len(tr_info['vl_aucs'])

    else:
        tr_info['tr_losses'].append(tr_loss)
        tr_info['tr_aucs'].append(ovft_metrics[0])
        tr_info['tr_baccs'].append(ovft_metrics[1])
        tr_info['tr_mccs'].append(ovft_metrics[2])

        tr_info['vl_losses'].append(vl_loss)
        tr_info['vl_aucs'].append(vl_metrics[0])
        tr_info['vl_baccs'].append(vl_metrics[1])
        tr_info['vl_mccs'].append(vl_metrics[2])


    return tr_info

def get_eval_string(tr_info, cycle):
    # pretty prints first three values of train/val metrics to a string and returns it
    s = 'Cycle {}: Tr/Vl AUC: {:.2f}/{:.2f} - bACC: {:.2f}/{:.2f} - MCC: {:.2f}/{:.2f} - Loss: {:.4f}/{:.4f}'.format(
        cycle+1, tr_info['tr_aucs'][cycle], tr_info['vl_aucs'][cycle], tr_info['tr_baccs'][cycle], tr_info['vl_baccs'][cycle],
              tr_info['tr_mccs'][cycle], tr_info['vl_mccs'][cycle], tr_info['tr_losses'][cycle], tr_info['vl_losses'][cycle])
    return s

def train_model(model, optimizer, criterion, train_loader, val_loader, overfit_loader, scheduler, save_path):

    n_cycles = len(scheduler.cycle_lens)
    best_metric, best_cycle = 0, 0
    tr_info = init_tr_info()

    for cycle in range(n_cycles):
        print('\nCycle {:d}/{:d}'.format(cycle+1, n_cycles))
        # train one cycle
        _, _, _, _ = train_one_cycle(model, optimizer, criterion, train_loader, scheduler, cycle=cycle)

        with torch.inference_mode():
            tr_probs, tr_preds, tr_labels, tr_loss = run_one_epoch(model, None, criterion, overfit_loader, assess=True)
            vl_probs, vl_preds, vl_labels, vl_loss = run_one_epoch(model, None, criterion, val_loader, assess=True)
        # print(vl_labels[:3], vl_labels[-3:])
        # print(vl_preds[:3], vl_preds[-3:])
        # print(vl_probs[:3], vl_probs[-3:])

        # sys.exit()
        ovft_metrics = get_class_metrics(tr_probs, tr_preds, tr_labels)
        vl_metrics = get_class_metrics(vl_probs, vl_preds, vl_labels)

        tr_info = set_tr_info(tr_info, vl_metrics, ovft_metrics, tr_loss=tr_loss, vl_loss=vl_loss)

        s = get_eval_string(tr_info, cycle)
        print(s)
        with open(osp.join(save_path, 'train_log.txt'), 'a') as f: print(s, file=f)

        # check if performance was better than anyone before and checkpoint if so
        if vl_metrics[0] > best_metric:
            print('-------- Best metric attained. {:.2f} --> {:.2f} --------'.format(best_metric, vl_metrics[0]))
            best_metric = vl_metrics[0]
            best_cycle = cycle+1
            tr_info = set_tr_info(tr_info, vl_metrics, cycle, best_cycle=True)
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            print('-------- Best metric so far {:.2f} at cycle {:d} --------'.format(best_metric, best_cycle))

    del model
    torch.cuda.empty_cache()
    return tr_info

def main(args):
    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        # print('Memory Usage:')
        # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        # print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    # reproducibility
    set_seed(args.seed, use_cuda)

    save_path = args.save_path
    if save_path is not None:
        save_path=osp.join('experiments_class', save_path)
        args.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        config_file_path = osp.join(save_path,'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)


    train_loader, val_loader, overfit_loader = get_train_val_class_dataloaders(args.fold, args.projection, args.batch_size, args.num_workers,
                                                                               cache = args.cache, persist=args.persist, proportion=args.proportion)

    # Prepare model for training
    model = get_arch(args.model, n_classes=1, pretrained=args.pretrained)
    model.to(device)

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
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise RuntimeError('Invalid optimizer {}. Only SGD and AdamW are supported.'.format(args.opt))


    cycle_lens = args.cycle_lens.split('/')
    cycle_lens = list(map(int, cycle_lens))
    if len(cycle_lens) > 2:
        sys.exit('cycles should be specified as a pair n_cycles/cycle_len')
    cycle_lens = cycle_lens[0] * [cycle_lens[1]]

    if args.sam:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=cycle_lens[0]*len(train_loader), eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle_lens[0]*len(train_loader), eta_min=0)
    setattr(optimizer, 'max_lr', args.lr)  # store maximum lr inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)


    if args.loss == 'ce':
        criterion = torch.nn.BCEWithLogitsLoss()
    else: sys.exit('what loss?')

    # Start training
    start = time.time()
    tr_info = train_model(model, optimizer, criterion, train_loader, val_loader, overfit_loader, scheduler, save_path)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('done')

    with (open(osp.join(save_path, 'log.txt'), 'w') as f):
        print('Best cycle = {}/{}\nBest Tr/Vl AUC = {:.2f}/{:.2f} - bACC = {:.2f}/{:.2f} '\
                '- MCC: {:.2f}/{:.2f} - Loss: {:.4f}/{:.4f}\n'.format(tr_info['best_cycle'], len(cycle_lens),
                tr_info['best_tr_auc'], tr_info['best_vl_auc'], tr_info['best_tr_bacc'], tr_info['best_vl_bacc'],
                tr_info['best_tr_mcc'], tr_info['best_vl_mcc'], tr_info['best_tr_loss'], tr_info['best_vl_loss']), file=f)

        tr_aucs, vl_aucs, tr_baccs, vl_baccs, tr_mccs, vl_mccs, tr_losses, vl_losses = tr_info['tr_aucs'], tr_info['vl_aucs'],  \
                    tr_info['tr_baccs'], tr_info['vl_baccs'], tr_info['tr_mccs'], tr_info['vl_mccs'], tr_info['tr_losses'], tr_info['vl_losses']
        for j in range(len(vl_aucs)):
            print('Cycle {}: Tr/Vl AUC: {:.2f}/{:.2f} - bACC: {:.2f}/{:.2f} - MCC: {:.2f}/{:.2f} - Loss: {:.4f}/{:.4f}'.format(j+1,
                   tr_aucs[j], vl_aucs[j], tr_baccs[j], vl_baccs[j], tr_mccs[j], vl_mccs[j], tr_losses[j], vl_losses[j]), file=f)
        print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)

    print('Done. Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
