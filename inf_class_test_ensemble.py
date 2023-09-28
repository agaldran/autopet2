from utils.model_class import get_arch
import os, os.path as osp
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

import monai.transforms as t
from monai.inferers import sliding_window_inference

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

    parser = argparse.ArgumentParser(description='Dataset Specifics')
    parser.add_argument('--experiment_path', type=str, default='last_experiment')
    parser.add_argument('--device', type=str, default='cuda', help='Overrides automatic device selection to pick cpu')
    parser.add_argument('--tta', type=str2bool, nargs='?', const=True, default=False, help='do tta')
    args = parser.parse_args()
    return args

def main(args):
    # will save predictions for each fold PLUS predictions of the ensemble!
    # experiment_path should end in f*, and we pick up folders f1,...,f5
    # e.g. experiments_class/mip_P_swin_wd3_f1

    # experiment_path: expect experiment_path = 'experiments_class/mip_P_swin_wd3_f'
    # python inf_class_test_ensemble.py --experiment_path experiments_class/mip_P_swin_wd3/
    save_path = osp.join('results_class', 'swin_wd3_ensemble')
    os.makedirs(save_path, exist_ok=True)
    if args.tta: save_name = osp.join(save_path, 'class_results_tta.npy')
    else: save_name = osp.join(save_path, 'class_results.npy')
    print('Saving to {}'.format(save_name))

    experiment_path_x_f1 = 'experiments_class/mipx_swin_wd3_f1/'
    experiment_path_x_f2 = 'experiments_class/mipx_swin_wd3_f2/'
    experiment_path_x_f3 = 'experiments_class/mipx_swin_wd3_f3/'
    experiment_path_x_f4 = 'experiments_class/mipx_swin_wd3_f4/'
    experiment_path_x_f5 = 'experiments_class/mipx_swin_wd3_f5/'
    experiment_path_x_list = [experiment_path_x_f1, experiment_path_x_f2, experiment_path_x_f3,
                               experiment_path_x_f4, experiment_path_x_f5]

    experiment_path_y_f1 = 'experiments_class/mipy_swin_wd3_f1/'
    experiment_path_y_f2 = 'experiments_class/mipy_swin_wd3_f2/'
    experiment_path_y_f3 = 'experiments_class/mipy_swin_wd3_f3/'
    experiment_path_y_f4 = 'experiments_class/mipy_swin_wd3_f4/'
    experiment_path_y_f5 = 'experiments_class/mipy_swin_wd3_f5/'
    experiment_path_y_list = [experiment_path_y_f1, experiment_path_y_f2, experiment_path_y_f3,
                              experiment_path_y_f4, experiment_path_y_f5]

    model_paths_x = [osp.join(e, 'best_model.pth') for e in experiment_path_x_list]
    model_paths_y = [osp.join(e, 'best_model.pth') for e in experiment_path_y_list]


    if args.device != 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    else: device = torch.device('cpu')

    model_x_list = [get_arch(model_name='swin', n_classes=1, pretrained=False) for _ in [0, 1, 2, 3, 4]]
    model_y_list = [get_arch(model_name='swin', n_classes=1, pretrained=False) for _ in [0, 1, 2, 3, 4]]

    for i in [0, 1, 2, 3, 4]:
        model_x_list[i].to(device)
        model_x_list[i].load_state_dict(torch.load(model_paths_x[i], map_location=device))
        model_x_list[i].eval()

        model_y_list[i].to(device)
        model_y_list[i].load_state_dict(torch.load(model_paths_y[i], map_location=device))
        model_y_list[i].eval()

    test_pos_data_dict = pd.read_csv('data/test_pos.csv', index_col=None).to_dict('records')
    test_neg_data_dict = pd.read_csv('data/test_neg.csv', index_col=None).to_dict('records')

    target_spacing = 2.04, 2.04, 3.00
    suv_p05, suv_p95 = 1.14, 30.17

    base_transforms = t.Compose([
        t.LoadImageD(keys='suv', image_only=True, ensure_channel_first=True),
        t.Orientationd(keys='suv', axcodes='RAS'),
        t.CropForegroundD(keys='suv', source_key='suv', select_fn=lambda suv: suv > suv_p05),
        t.CropForegroundD(keys='suv', source_key='suv', select_fn=lambda suv: suv > suv_p05),
        t.Spacingd(keys='suv', pixdim=target_spacing, mode=('bilinear',)),
        t.ThresholdIntensityD(keys='suv', above=False, threshold=suv_p95, cval=suv_p95),
        t.ThresholdIntensityD(keys='suv', above=True, threshold=suv_p05, cval=suv_p05),
    ])

    from utils.data_load import get_mip_transforms


    flip_X = t.Flipd(keys='suv', spatial_axis=-3)
    flip_Y = t.Flipd(keys='suv', spatial_axis=-2)

    all_probs_x, all_probs_y = [], []
    with torch.no_grad():
        print('* Test Positive Samples')
        for i in tqdm(range(len(test_pos_data_dict))):
            scan = base_transforms(test_pos_data_dict[i])
            # Process X-projection
            inputs_x = get_mip_transforms('x')(scan)['suv'].unsqueeze(0)
            prob_x_list = []
            for model_x in model_x_list:
                prob_x = model_x(inputs_x.to(device)).sigmoid().item()
                if args.tta:
                    inputs_x = get_mip_transforms('x')(flip_Y(scan))['suv'].unsqueeze(0)
                    prob_x += model_x(inputs_x.to(device)).sigmoid().item()
                    prob_x /= 2
                prob_x_list.append(prob_x)
            prob_x = np.mean(prob_x_list)

            # Process Y-projection
            inputs_y = get_mip_transforms('y')(scan)['suv'].unsqueeze(0)
            prob_y_list = []
            for model_y in model_y_list:
                prob_y = model_y(inputs_y.to(device)).sigmoid().item()
                if args.tta:
                    inputs_y = get_mip_transforms('y')(flip_X(scan))['suv'].unsqueeze(0)
                    prob_y += model_y(inputs_y.to(device)).sigmoid().item()
                    prob_y /= 2
                prob_y_list.append(prob_y)
            prob_y = np.mean(prob_y_list)

            all_probs_x.append(prob_x)
            all_probs_y.append(prob_y)

        print('* Test Negative Samples')
        for i in tqdm(range(len(test_neg_data_dict))):
            scan = base_transforms(test_neg_data_dict[i])
            # Process X-projection
            inputs_x = get_mip_transforms('x')(scan)['suv'].unsqueeze(0)
            prob_x_list = []
            for model_x in model_x_list:
                prob_x = model_x(inputs_x.to(device)).sigmoid().item()
                if args.tta:
                    inputs_x = get_mip_transforms('x')(flip_Y(scan))['suv'].unsqueeze(0)
                    prob_x += model_x(inputs_x.to(device)).sigmoid().item()
                    prob_x /= 2
                prob_x_list.append(prob_x)
            prob_x = np.mean(prob_x_list)

            # Process Y-projection
            inputs_y = get_mip_transforms('y')(scan)['suv'].unsqueeze(0)
            prob_y_list = []
            for model_y in model_y_list:
                prob_y = model_y(inputs_y.to(device)).sigmoid().item()
                if args.tta:
                    inputs_y = get_mip_transforms('y')(flip_X(scan))['suv'].unsqueeze(0)
                    prob_y += model_y(inputs_y.to(device)).sigmoid().item()
                    prob_y /= 2
                prob_y_list.append(prob_y)
            prob_y = np.mean(prob_y_list)

            all_probs_x.append(prob_x)
            all_probs_y.append(prob_y)

    labels = len(test_pos_data_dict)*[1] + len(test_neg_data_dict)*[0]
    np.save(save_name, np.stack([np.array(all_probs_x), np.array(all_probs_y), np.array(labels)], axis=0))

if __name__ == "__main__":
    args = get_args_parser()
    main(args)