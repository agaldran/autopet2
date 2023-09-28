"""
Contains implementations of transforms and dataloaders needed for training, validation and inference.
"""
import os, sys, os.path as osp
import numpy as np
import pandas as pd
from monai.data import Dataset, CacheDataset, DataLoader, PersistentDataset
import torch.nn as nn
import monai.transforms as t


#################### CLASSIFICATION DATA HANDLING ####################
def mipx(img): return img.max(dim=-3)[0]

def mipy(img): return img.max(dim=-2)[0]

def mipz(img): return img.max(dim=-1)[0]

def get_mip_transforms(projection):
    mipx_shape = (224, 352)
    mipy_shape = (224, 352)
    mipz_shape = (224, 224)

    if projection == 'x':
        mip_func, spatial_size = mipx, mipx_shape
    elif projection == 'y':
        mip_func, spatial_size = mipy, mipy_shape
    elif projection == 'z':
        mip_func, spatial_size = mipz, mipz_shape

    mip_transforms = t.Compose([
        t.Lambdad(keys=('suv'), func=mip_func),
        t.ResizeD(keys=('suv'), spatial_size=spatial_size),
        t.NormalizeIntensityD(keys=('suv'), channel_wise=True, nonzero=True),
        t.RepeatChannelD(keys=('suv'), repeats=3)
    ])

    return mip_transforms

def get_train_val_class_transforms(projection):
    assert projection in ['x', 'y', 'z']
    target_spacing = 2.04, 2.04, 3.00
    suv_p05, suv_p95 = 1.14, 30.17
    p_app, pr_geom = 0.5, 0.5

    base_transforms = t.Compose([
        t.LoadImageD(keys=('suv'), image_only=True, ensure_channel_first=True),
        t.Orientationd(keys=('suv'), axcodes='RAS'),
        t.CropForegroundD(keys=('suv'), source_key='suv', select_fn=lambda suv: suv > suv_p05),
        t.CropForegroundD(keys=('suv'), source_key='suv', select_fn=lambda suv: suv > suv_p05),
        t.Spacingd(keys=('suv'), pixdim=target_spacing, mode=('bilinear',)),
        t.ThresholdIntensityD(keys=('suv'), above=False, threshold=suv_p95, cval=suv_p95),
        t.ThresholdIntensityD(keys=('suv'), above=True, threshold=suv_p05, cval=suv_p05),
    ])
    # for richer augmentation, these must happen before projection
    data_augmentation = t.Compose([
        t.RandScaleIntensityD(keys=('suv'), factors=0.10, prob=p_app),
        t.RandShiftIntensityd(keys=('suv'), offsets=0.10, prob=p_app),
        t.RandRotated(keys=('suv'), range_x=(np.deg2rad(-10), np.deg2rad(10)), range_y=(np.deg2rad(-10), np.deg2rad(10)),
                       range_z=(np.deg2rad(-10), np.deg2rad(10)), prob=pr_geom, mode=('bilinear',)),
        t.RandZoomd(keys=('suv'), min_zoom=0.95, max_zoom=1.05, prob=pr_geom, mode=('bilinear',)),
        t.RandFlipd(('suv'), spatial_axis=[0], prob=pr_geom),
        t.RandFlipd(('suv'), spatial_axis=[1], prob=pr_geom),
        t.RandFlipd(('suv'), spatial_axis=[2], prob=pr_geom),
    ])
    tr_transforms = t.Compose([base_transforms, data_augmentation, get_mip_transforms(projection)])
    vl_transforms = t.Compose([base_transforms, get_mip_transforms(projection)])

    return tr_transforms, vl_transforms

def get_train_val_class_dataloaders(fold, projection, batch_size=8, num_workers=0, cache=0., persist=False, proportion=1 ):
    train_pos_data_dict = pd.read_csv('data/train_pos_f{}.csv'.format(fold), index_col=None).to_dict('records')
    train_neg_data_dict = pd.read_csv('data/train_neg_f{}.csv'.format(fold), index_col=None).to_dict('records')
    val_pos_data_dict = pd.read_csv('data/val_pos_f{}.csv'.format(fold), index_col=None).to_dict('records')
    val_neg_data_dict = pd.read_csv('data/val_neg_f{}.csv'.format(fold), index_col=None).to_dict('records')

    train_data_dict = train_pos_data_dict + train_neg_data_dict
    val_data_dict = val_pos_data_dict + val_neg_data_dict

    import random
    random.shuffle(train_data_dict)
    random.shuffle(val_data_dict)

    if proportion < 1.:
        train_data_dict = train_data_dict[: int(proportion * len(train_data_dict))]
        val_data_dict = val_data_dict[: int(proportion * len(val_data_dict))]

    tr_transforms, vl_transforms = get_train_val_class_transforms(projection)

    if persist:
        persistent_cache = osp.join('data', 'persistent_cache_class_p{}'.format(projection))
        tr_ds = PersistentDataset(data=train_data_dict, transform=tr_transforms, cache_dir=persistent_cache)
        vl_ds = PersistentDataset(data=val_data_dict, transform=vl_transforms, cache_dir=persistent_cache)
        overfit_ds = PersistentDataset(data=train_data_dict[:20], transform=vl_transforms, cache_dir=persistent_cache)

    else:
        tr_ds = CacheDataset(data=train_data_dict, transform=tr_transforms, cache_rate=cache, num_workers=num_workers)
        vl_ds = CacheDataset(data=val_data_dict, transform=vl_transforms, cache_rate=cache, num_workers=num_workers)
        overfit_ds = CacheDataset(data=train_data_dict[:20], transform=vl_transforms, cache_rate=cache, num_workers=num_workers)

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    vl_dl = DataLoader(vl_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    overfit_dl = DataLoader(overfit_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    _, b = np.unique(np.array([v['positive'] for v in tr_dl.dataset.data]), return_counts=True)
    print('TRAIN: Negative = {}, Positive = {}'.format(b[0], b[1]))
    _, b = np.unique(np.array([v['positive'] for v in vl_dl.dataset.data]), return_counts=True)
    print('VALIDATION: Negative = {}, Positive = {}'.format(b[0], b[1]))
    _, b = np.unique(np.array([v['positive'] for v in overfit_dl.dataset.data]), return_counts=True)
    print('OVERFIT: Negative = {}, Positive = {}'.format(b[0], b[1]))

    return tr_dl, vl_dl, overfit_dl


#################### SEGMENTATION DATA HANDLING ####################
def get_train_val_transforms(n_samples, neg_samples, patch_size):
    target_spacing = 2.04, 2.04, 3.00
    ct_p05, ct_p95 = -799.20, 557.91
    suv_p05, suv_p95 = 1.14, 30.17
    p_app, pr_geom = 0.1, 0.1

    tr_transforms = t.Compose([
        t.LoadImageD(keys=('ct', 'suv', 'seg'), ensure_channel_first=True, image_only=True),
        t.CropForegroundD(keys=('ct', 'suv', 'seg'), source_key='ct', select_fn=lambda ct: ct > ct_p05),
        t.Orientationd(keys=('ct', 'suv', 'seg'), axcodes='RAS'),
        t.Spacingd(keys=('ct', 'suv', 'seg'), pixdim=target_spacing, mode=('bilinear', 'bilinear', 'nearest')),
        t.ThresholdIntensityD(keys=('ct'), above=False, threshold=ct_p95, cval=ct_p95),
        t.ThresholdIntensityD(keys=('ct'), above=True, threshold=ct_p05, cval=ct_p05),
        t.ThresholdIntensityD(keys=('suv'), above=False, threshold=suv_p95, cval=suv_p95),
        t.ThresholdIntensityD(keys=('suv'), above=True, threshold=suv_p05, cval=suv_p05),

        t.NormalizeIntensityD(keys=('ct', 'suv'), channel_wise=True, nonzero=True),
        t.RandCropByPosNegLabeld(keys=('ct', 'suv', 'seg'), label_key='seg', spatial_size=patch_size, num_samples=n_samples,
                                 pos=1, neg=neg_samples),  # P(center=fground) = pos/(pos+neg) = 1/(1+neg)
        # t.RandCropByPosNegLabeld(keys=('ct', 'suv', 'seg'), label_key='seg', spatial_size=patch_size, allow_smaller=True,
        #                          num_samples=n_samples, pos=1, neg=neg_samples),  # P(center=fground) = pos/(pos+neg) = 1/(1+neg)
        # t.SpatialPadD(keys=('ct', 'suv', 'seg'), spatial_size=patch_size),
        t.RandScaleIntensityD(keys=('ct', 'suv'), factors=0.05, prob=p_app),
        t.RandShiftIntensityd(keys=('ct', 'suv'), offsets=0.05, prob=p_app),
        t.RandRotated(keys=('ct', 'suv', 'seg'), range_x=(np.deg2rad(-10), np.deg2rad(10)), range_y=(np.deg2rad(-10), np.deg2rad(10)),
                      range_z=(np.deg2rad(-10), np.deg2rad(10)), prob=pr_geom, mode=('bilinear', 'bilinear', 'nearest')),
        t.RandZoomd(keys=('ct', 'suv', 'seg'), min_zoom=0.95, max_zoom=1.05, prob=pr_geom, mode=('bilinear', 'bilinear', 'nearest')),
        t.RandFlipd(('ct', 'suv', 'seg'), spatial_axis=[0], prob=pr_geom),
        t.RandFlipd(('ct', 'suv', 'seg'), spatial_axis=[1], prob=pr_geom),
        t.RandFlipd(('ct', 'suv', 'seg'), spatial_axis=[2], prob=pr_geom),

        t.ConcatItemsd(keys=('ct', 'suv'), name='ct_suv', dim=0),
        t.DeleteItemsd(keys=('ct', 'suv')),
        # t.CastToTyped(keys=('ct_suv', 'seg'), dtype=(np.float32, np.uint8)),
    ])

    vl_transforms = t.Compose([
        t.LoadImageD(keys=('ct', 'suv', 'seg'), ensure_channel_first=True, image_only=True),
        t.CropForegroundD(keys=('ct', 'suv', 'seg'), source_key='ct', select_fn=lambda ct: ct > ct_p05),
        t.Orientationd(keys=('ct', 'suv', 'seg'), axcodes='RAS'),
        t.Spacingd(keys=('ct', 'suv', 'seg'), pixdim=target_spacing, mode=('bilinear', 'bilinear', 'nearest')),
        t.ThresholdIntensityD(keys=('ct'), above=False, threshold=ct_p95, cval=ct_p95),
        t.ThresholdIntensityD(keys=('ct'), above=True, threshold=ct_p05, cval=ct_p05),
        t.ThresholdIntensityD(keys=('suv'), above=False, threshold=suv_p95, cval=suv_p95),
        t.ThresholdIntensityD(keys=('suv'), above=True, threshold=suv_p05, cval=suv_p05),
        t.NormalizeIntensityD(keys=('ct', 'suv'), channel_wise=True, nonzero=True),
        t.ConcatItemsd(keys=('ct', 'suv'), name='ct_suv', dim=0),
        t.DeleteItemsd(keys=('ct', 'suv')),
        # t.CastToTyped(keys=('ct_suv', 'seg'), dtype=(np.float32, np.uint8)),
    ])

    return tr_transforms, vl_transforms

def get_train_val_dataloaders(fold, n_samples, neg_samples, patch_size, proportion=1, num_workers=0, cache=0., persist=False, ):
    train_data_dict = pd.read_csv('data/train_f{}.csv'.format(fold), index_col=None).to_dict('records')
    val_pos_data_dict = pd.read_csv('data/val_pos_f{}.csv'.format(fold), index_col=None).to_dict('records')
    val_neg_data_dict = pd.read_csv('data/val_neg_f{}.csv'.format(fold), index_col=None).to_dict('records')

    if proportion < 1.:
        train_data_dict = train_data_dict[: int(proportion * len(train_data_dict))]
        val_pos_data_dict = val_pos_data_dict[: int(proportion * len(val_pos_data_dict))]
        val_neg_data_dict = val_neg_data_dict[: int(proportion * len(val_neg_data_dict))]

    tr_transforms, vl_transforms = get_train_val_transforms(n_samples, neg_samples, patch_size)

    if persist:
        persistent_cache = osp.join('data', 'persistent_cache')
        tr_ds = PersistentDataset(data=train_data_dict, transform=tr_transforms, cache_dir=persistent_cache)
        vl_pos_ds = PersistentDataset(data=val_pos_data_dict, transform=vl_transforms, cache_dir=persistent_cache)
        vl_neg_ds = PersistentDataset(data=val_neg_data_dict, transform=vl_transforms, cache_dir=persistent_cache)
    else:
        tr_ds = CacheDataset(data=train_data_dict, transform=tr_transforms, cache_rate=cache, num_workers=4)
        vl_pos_ds = CacheDataset(data=val_pos_data_dict, transform=vl_transforms, cache_rate=cache, num_workers=4)
        vl_neg_ds = CacheDataset(data=val_neg_data_dict, transform=vl_transforms, cache_rate=cache, num_workers=4)

    tr_dl = DataLoader(tr_ds, batch_size=1, shuffle=True, num_workers=num_workers)
    vl_pos_dl = DataLoader(vl_pos_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    vl_neg_dl = DataLoader(vl_neg_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return tr_dl, vl_pos_dl, vl_neg_dl

def get_train_val_pos_dataloaders(fold, n_samples, neg_samples, patch_size, proportion=1, notest=False, num_workers=0, cache=0., persist=False, ):
    train_data_dict = pd.read_csv('data/train_pos_f{}.csv'.format(fold), index_col=None).to_dict('records')
    val_data_dict = pd.read_csv('data/val_pos_f{}.csv'.format(fold), index_col=None).to_dict('records')
    if notest:
        train_data_dict = pd.read_csv('data/train_pos_notest_f{}.csv'.format(fold), index_col=None).to_dict('records')

    if proportion < 1.:
        train_data_dict = train_data_dict[: int(proportion * len(train_data_dict))]
        val_data_dict = val_data_dict[: int(proportion * len(val_data_dict))]

    tr_transforms, vl_transforms = get_train_val_transforms(n_samples, neg_samples, patch_size)

    if persist:
        persistent_cache = osp.join('data', 'persistent_cache')
        tr_ds = PersistentDataset(data=train_data_dict, transform=tr_transforms, cache_dir=persistent_cache)
        vl_ds = PersistentDataset(data=val_data_dict, transform=vl_transforms, cache_dir=persistent_cache)
        overfit_ds = PersistentDataset(data=train_data_dict[:20], transform=vl_transforms, cache_dir=persistent_cache)
    else:
        tr_ds = CacheDataset(data=train_data_dict, transform=tr_transforms, cache_rate=cache, num_workers=0)
        vl_ds = CacheDataset(data=val_data_dict, transform=vl_transforms, cache_rate=cache, num_workers=0)
        overfit_ds = CacheDataset(data=train_data_dict[:20], transform=vl_transforms, cache_rate=cache, num_workers=0)

    tr_dl = DataLoader(tr_ds, batch_size=1, shuffle=True, num_workers=num_workers)
    vl_dl = DataLoader(vl_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    overfit_dl = DataLoader(overfit_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return tr_dl, vl_dl, overfit_dl

if __name__ == "__main__":

    print('pass')
