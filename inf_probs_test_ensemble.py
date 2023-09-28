from utils.model import get_model
import os, os.path as osp
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

import monai.transforms as t
from monai.inferers import sliding_window_inference

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Dataset Specifics')
    parser.add_argument('--experiment_path', type=str, default='last_experiment')
    parser.add_argument('--device', type=str, default='cuda', help='Overrides automatic device selection to pick cpu')
    parser.add_argument('--tta', type=int, default=0, help='tta')
    args = parser.parse_args()
    return args

def main(args):
    # will save predictions for each fold PLUS predictions of the ensemble!
    # experiment_path should end in F*, and we pick up folders F1,...,F5
    # e.g. experiments/swin_tiny_F
    # experiment_path = osp.join('experiments', args.experiment_path)

    if args.device != 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = 'cpu'
    save_path = osp.join('results', args.experiment_path.split('/')[-1].replace('_F', '_ENSEMBLE'))

    path_model_state_F1 = args.experiment_path+'1/best_model.pth'
    path_model_state_F2 = args.experiment_path+'2/best_model.pth'
    path_model_state_F3 = args.experiment_path+'3/best_model.pth'
    path_model_state_F4 = args.experiment_path+'4/best_model.pth'
    path_model_state_F5 = args.experiment_path+'5/best_model.pth'

    patch_size, slwin_batch_size = (96, 96, 96), 4

    model_f1 = get_model('swinunetr_tiny', patch_size).to(device)
    model_f1.load_state_dict(torch.load(path_model_state_F1, map_location=device))
    model_f1.eval()

    model_f2 = get_model('swinunetr_tiny', patch_size).to(device)
    model_f2.load_state_dict(torch.load(path_model_state_F2, map_location=device))
    model_f2.eval()

    model_f3 = get_model('swinunetr_tiny', patch_size).to(device)
    model_f3.load_state_dict(torch.load(path_model_state_F3, map_location=device))
    model_f3.eval()

    model_f4 = get_model('swinunetr_tiny', patch_size).to(device)
    model_f4.load_state_dict(torch.load(path_model_state_F4, map_location=device))
    model_f4.eval()

    model_f5 = get_model('swinunetr_tiny', patch_size).to(device)
    model_f5.load_state_dict(torch.load(path_model_state_F5, map_location=device))
    model_f5.eval()

    seg_model_list = [model_f1, model_f2, model_f3, model_f4, model_f5]



    test_pos_data_dict = pd.read_csv('data/test_pos.csv', index_col=None).to_dict('records')
    test_neg_data_dict = pd.read_csv('data/test_neg.csv', index_col=None).to_dict('records')

    target_spacing = 2.04, 2.04, 3.00
    ct_p05, ct_p95 = -799.20, 557.91
    suv_p05, suv_p95 = 1.14, 30.17

    base_transforms = t.Compose([t.LoadImageD(keys=('ct','suv','seg'), ensure_channel_first=True, image_only=False),
                        t.CropForegroundD(keys=('ct','suv','seg'), source_key='ct', select_fn=lambda ct:ct>ct_p05),
                        t.Orientationd(keys=('ct', 'suv','seg'), axcodes='RAS'),
                        t.Spacingd(keys=('ct', 'suv','seg'), pixdim=target_spacing,
                                   mode=('bilinear', 'bilinear', 'nearest'))])

    test_transforms = t.Compose([
        t.ThresholdIntensityD(keys=('ct'), above=False, threshold=ct_p95, cval=ct_p95),
        t.ThresholdIntensityD(keys=('ct'), above=True, threshold=ct_p05, cval=ct_p05),
        t.ThresholdIntensityD(keys=('suv'), above=False, threshold=suv_p95, cval=suv_p95),
        t.ThresholdIntensityD(keys=('suv'), above=True, threshold=suv_p05, cval=suv_p05),
        t.NormalizeIntensityD(keys=('ct', 'suv'), channel_wise=True, nonzero=True),])

    print('* Validation Positive Samples')
    for i in tqdm(range(len(test_pos_data_dict))):
        x = base_transforms(test_pos_data_dict[i])
        x = test_transforms(x)
        with (torch.no_grad()):
            inputs = torch.cat([x['ct'], x['suv']]).unsqueeze(0).to(device)
            acc_seg = []
            for model in seg_model_list:
                x_out = sliding_window_inference(inputs, patch_size, slwin_batch_size, model, mode='gaussian').sigmoid()

                if args.tta == 1:
                    x_out += sliding_window_inference(inputs.flip(-1), patch_size, slwin_batch_size, model, mode='gaussian').flip(-1).sigmoid()
                    x_out /= 2
                elif args.tta == 2:
                    x_out += sliding_window_inference(inputs.flip(-1), patch_size, slwin_batch_size, model, mode='gaussian').flip(-1).sigmoid()
                    x_out += sliding_window_inference(inputs.flip(-2), patch_size, slwin_batch_size, model, mode='gaussian').flip(-2).sigmoid()
                    x_out /= 3
                elif args.tta == 3:
                    x_out += sliding_window_inference(inputs.flip(-1), patch_size, slwin_batch_size, model, mode='gaussian').flip(-1).sigmoid()
                    x_out += sliding_window_inference(inputs.flip(-2), patch_size, slwin_batch_size, model, mode='gaussian').flip(-2).sigmoid()
                    x_out += sliding_window_inference(inputs.flip(-3), patch_size, slwin_batch_size, model, mode='gaussian').flip(-3).sigmoid()
                    x_out /= 4
                else:
                    if args.tta != 0: import sys; sys.exit('Max number of augmentations is 3')

                acc_seg.append(x_out[0])
        # print(x_out[0].shape)
        # print(torch.stack(acc_seg, dim=0).shape)
        # print(torch.mean(torch.stack(acc_seg, dim=0), dim=0).shape)
        # import sys; sys.exit()
        x['ct'] = torch.mean(torch.stack(acc_seg, dim=0), dim=0)
        y = t.Invertd(keys=('ct', 'seg'), transform=base_transforms, orig_keys=('ct', 'seg'), )(x)

        # save ints
        output_dir = osp.join(save_path, 'tta{}_test/positives'.format(args.tta))
        t.SaveImage(output_dir=output_dir, separate_folder=False, resample=False, output_postfix='',
                    print_log=False)((256*y['ct']).astype(np.uint8), meta_data=x['ct_meta_dict'])

    print('* Validation Negative Samples')
    for i in tqdm(range(len(test_neg_data_dict))):
        x = base_transforms(test_neg_data_dict[i])
        x = test_transforms(x)
        with (torch.no_grad()):
            inputs = torch.cat([x['ct'], x['suv']]).unsqueeze(0).to(device)
            acc_seg = []
            for model in seg_model_list:
                x_out = sliding_window_inference(inputs, patch_size, slwin_batch_size, model, mode='gaussian').sigmoid()

                if args.tta == 1:
                    x_out += sliding_window_inference(inputs.flip(-1), patch_size, slwin_batch_size, model, mode='gaussian').flip(-1).sigmoid()
                    x_out /= 2
                elif args.tta == 2:
                    x_out += sliding_window_inference(inputs.flip(-1), patch_size, slwin_batch_size, model, mode='gaussian').flip(-1).sigmoid()
                    x_out += sliding_window_inference(inputs.flip(-2), patch_size, slwin_batch_size, model, mode='gaussian').flip(-2).sigmoid()
                    x_out /= 3
                elif args.tta == 3:
                    x_out += sliding_window_inference(inputs.flip(-1), patch_size, slwin_batch_size, model, mode='gaussian').flip(-1).sigmoid()
                    x_out += sliding_window_inference(inputs.flip(-2), patch_size, slwin_batch_size, model, mode='gaussian').flip(-2).sigmoid()
                    x_out += sliding_window_inference(inputs.flip(-3), patch_size, slwin_batch_size, model, mode='gaussian').flip(-3).sigmoid()
                    x_out /= 4
                else:
                    if args.tta != 0: import sys; sys.exit('Max number of augmentations is 3')

                acc_seg.append(x_out[0])

        x['ct'] = torch.mean(torch.stack(acc_seg))
        y = t.Invertd(keys=('ct', 'seg'), transform=base_transforms, orig_keys=('ct', 'seg'), )(x)

        # # save floats
        # t.SaveImage(output_dir=osp.join('results', model_path.split('/')[1], 'negatives/probs'), separate_folder=False,
        #             resample=False, output_postfix='', print_log=False)(y['ct'], meta_data=x['ct_meta_dict'])

        # save ints
        output_dir = osp.join(save_path, 'tta{}_test/negatives'.format(args.tta))
        t.SaveImage(output_dir=output_dir, separate_folder=False, resample=False, output_postfix='',
                    print_log=False)((256 * y['ct']).astype(np.uint8), meta_data=x['ct_meta_dict'])



if __name__ == "__main__":
    args = get_args_parser()
    main(args)