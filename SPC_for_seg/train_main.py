import json

from SPC2.SPC2.test import test
from validation import *
from test import *
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
from FedAvg import FedAvg, aggregation
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from networks.models import DenseNet121
from dataloaders import dataset
from local_supervised import SupervisedLocalUpdate
from local_unsupervised import UnsupervisedLocalUpdate
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tsne.picture_all import read_data_tsne
from tqdm import tqdm
import shutil
import argparse
import logging
import random
import numpy as np
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.Unet3d import BaselineUNet
from dataloaders.dataset import *
from monai.losses import DiceCELoss

from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    PersistentDataset,
    decollate_batch,
)

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotated,
    ToTensord,
    PadListDataCollate,
)


# 训练测试
def Test(save_mode_path, epoch):
    checkpoint_path = save_mode_path
    checkpoint = torch.load(checkpoint_path)

    net = BaselineUNet(in_channels=1, n_cls=num_classes, n_filters=24)
    model = torch.nn.DataParallel(net, device_ids=device_ids).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    logging.info("--------------------------Test--------------------------")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=True)

    epoch_iterator_test = tqdm(test_loader, dynamic_ncols=True)
    dice_all, hd_all, asd_all, nsd_all, jc_all = test(epoch_iterator_test, model)

    dice_avg = (dice_all[0] + dice_all[1]) / 2
    hd_avg = (hd_all[0] + hd_all[1]) / 2
    asd_avg = (asd_all[0] + asd_all[1]) / 2
    nsd_avg = (nsd_all[0] + nsd_all[1]) / 2
    jc_avg = (jc_all[0] + jc_all[1]) / 2

    logging.info('Round:{:.4f} '
                'average DSC:{:.4f} ''DSC_tumor:{:.4f} ''DSC_lymph:{:.4f} '
                'average HD95:{:.4f} ''HD95_tumor:{:.4f} ''HD95_lymph:{:.4f} '
                'average asd:{:.4f} ''asd_tumor:{:.4f} ''asd_lymph:{:.4f} '
                'average nsd:{:.4f} ''nsd_tumor:{:.4f} ''nsd_lymph:{:.4f} '
                'average jc:{:.4f} ''jc_tumor:{:.4f} ''jc_lymph:{:.4f} '
                .format(epoch, dice_avg, dice_all[0], dice_all[1], hd_avg, hd_all[0], hd_all[1]
                        , asd_avg, asd_all[0], asd_all[1], nsd_avg, nsd_all[0], nsd_all[1], jc_avg, jc_all[0],
                        jc_all[1]))
    logging.info("--------------------------End Test--------------------------")

    return dice_avg, hd_avg, asd_avg, nsd_avg, jc_avg


# 训练验证
def Validate(save_mode_path, epoch):
    checkpoint_path = save_mode_path
    checkpoint = torch.load(checkpoint_path)

    net = BaselineUNet(in_channels=1, n_cls=num_classes, n_filters=24)
    model = torch.nn.DataParallel(net, device_ids=device_ids).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    logging.info("--------------------------Validation--------------------------")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=True)

    epoch_iterator_val = tqdm(val_loader, dynamic_ncols=True)
    dice_all, hd_all, asd_all, nsd_all, jc_all = validation(epoch_iterator_val, model)

    dice_avg = (dice_all[0] + dice_all[1]) / 2
    hd_avg = (hd_all[0] + hd_all[1]) / 2
    asd_avg = (asd_all[0] + asd_all[1]) / 2
    nsd_avg = (nsd_all[0] + nsd_all[1]) / 2
    jc_avg = (jc_all[0] + jc_all[1]) / 2

    logging.info('Round:{:.4f} '
                 'average DSC:{:.4f} ''DSC_tumor:{:.4f} ''DSC_lymph:{:.4f} '
                 'average HD95:{:.4f} ''HD95_tumor:{:.4f} ''HD95_lymph:{:.4f} '
                 'average asd:{:.4f} ''asd_tumor:{:.4f} ''asd_lymph:{:.4f} '
                 'average nsd:{:.4f} ''nsd_tumor:{:.4f} ''nsd_lymph:{:.4f} '
                 'average jc:{:.4f} ''jc_tumor:{:.4f} ''jc_lymph:{:.4f} '
                 .format(epoch, dice_avg, dice_all[0], dice_all[1], hd_avg, hd_all[0], hd_all[1]
                         , asd_avg, asd_all[0], asd_all[1], nsd_avg, nsd_all[0], nsd_all[1], jc_avg, jc_all[0],
                         jc_all[1]))
    logging.info("--------------------------End Validation--------------------------")

    return dice_avg, hd_avg, asd_avg, nsd_avg, jc_avg


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]

# 保存位置
snapshot_path = 'D:/Bai/Spc2/SPC2/SPC2/models/scenario3'

supervised_user_id = [0]
unsupervised_user_id = [1, 2]
begin_unsupervised = 450
dict_users = []
dict_length = []
com_round_interval = 10
center_avg = torch.zeros((3, 192)).to(device)
c_t_avg = torch.zeros((192)).to(device)
flag_create = False

if __name__ == '__main__':

    args = args_parser()
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.basicConfig(filename="D:/Bai/Spc2/SPC2/SPC2/models/scenario3/log.txt", level=logging.INFO, filemode="w",
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Initialize variables
    # Data
    ct_a_min = -430
    ct_a_max = 770
    strength = 1  # Data aug strength
    p = 0.5  # Data aug transforms probability

    num_classes = 3

    # Data transforms
    image_keys = ['img', 'mask']  # Do not change
    modes_3d = ['trilinear', 'nearest']
    modes_2d = ['bilinear', 'nearest']
    train_transforms = Compose(
        [
            LoadImaged(keys=image_keys),
            AddChanneld(keys=image_keys),
            Orientationd(keys=image_keys, axcodes='RAS'),
            ScaleIntensityRanged(
                keys=['img'],
                a_min=ct_a_min,
                a_max=ct_a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandAffined(keys=image_keys, prob=p,
                        translate_range=(round(10 * strength), round(10 * strength), round(10 * strength)),
                        padding_mode='border', mode=modes_2d),
            RandAffined(keys=image_keys, prob=p, scale_range=(0.10 * strength, 0.10 * strength, 0.10 * strength),
                        padding_mode='border', mode=modes_2d),
            RandFlipd(
                keys=image_keys,
                spatial_axis=[0],
                prob=p / 3,
            ),
            RandFlipd(
                keys=image_keys,
                spatial_axis=[1],
                prob=p / 3,
            ),
            RandFlipd(
                keys=image_keys,
                spatial_axis=[2],
                prob=p / 3,
            ),
            RandShiftIntensityd(
                keys=['img'],
                offsets=0.10,
                prob=p,
            ),
            ToTensord(keys=image_keys),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=image_keys),
            AddChanneld(keys=image_keys),
            Orientationd(keys=image_keys, axcodes='RAS'),
            ScaleIntensityRanged(keys=['img'], a_min=ct_a_min, a_max=ct_a_max, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=image_keys),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=image_keys),
            AddChanneld(keys=image_keys),
            Orientationd(keys=image_keys, axcodes='RAS'),
            ScaleIntensityRanged(keys=['img'], a_min=ct_a_min, a_max=ct_a_max, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=image_keys),
        ]
    )

    load_split_path = r"D:/Bai/Spc2/SPC2/SPC2/partition_strategy/CBCT/split_1.pkl"

    with open(load_split_path) as f:
        split_dict = json.load(f)

    labeled_list = split_dict['labeled']
    unlabeled_list = split_dict['unlabeled']
    val_list = split_dict['val']
    test_list = split_dict['test']

    labeled_img = [f"{patient_name}.nii.gz" for patient_name in labeled_list]
    labeled_mask = [f"{patient_name}_label.nii.gz" for patient_name in labeled_list]

    unlabeled_img_1 = [f"{patient_name}.nii.gz" for patient_name in unlabeled_list[0]]
    unlabeled_mask_1 = [f"{patient_name}_label.nii.gz" for patient_name in unlabeled_list[0]]

    unlabeled_img_2 = [f"{patient_name}.nii.gz" for patient_name in unlabeled_list[1]]
    unlabeled_mask_2 = [f"{patient_name}_label.nii.gz" for patient_name in unlabeled_list[1]]

    val_img = [f"{patient_name}.nii.gz" for patient_name in val_list]
    val_mask = [f"{patient_name}_label.nii.gz" for patient_name in val_list]

    test_img = [f"{patient_name}.nii.gz" for patient_name in test_list]
    test_mask = [f"{patient_name}_label.nii.gz" for patient_name in test_list]

    # Initialize DataLoader
    val_dict = [
        {'img': os.path.join(args.root_path, img), 'mask': os.path.join(args.root_path, mask)} for
        img, mask in zip(val_img, val_mask)]
    val_ds = CacheDataset(data=val_dict, transform=val_transforms, cache_rate=1.0, num_workers=0)

    test_dict = [
        {'img': os.path.join(args.root_path, img), 'mask': os.path.join(args.root_path, mask)} for
        img, mask in zip(test_img, test_mask)]
    test_ds = CacheDataset(data=test_dict, transform=test_transforms, cache_rate=1.0, num_workers=0)

    for i in range(args.num_users):
        if i == 0:
            labeled_dict = [
                {'img': os.path.join(args.root_path, img), 'mask': os.path.join(args.root_path, mask)} for
                img, mask in zip(labeled_img, labeled_mask)]
            train_dataset = CacheDataset(data=labeled_dict, transform=train_transforms, cache_rate=1.0, num_workers=0)
        elif i == 1:
            unlabeled_dict = [
                {'img': os.path.join(args.root_path, img), 'mask': os.path.join(args.root_path, mask)} for
                img, mask in zip(unlabeled_img_1, unlabeled_mask_1)]
            train_dataset = CacheDataset(data=unlabeled_dict, transform=train_transforms, cache_rate=1.0, num_workers=0)
        else:
            unlabeled_dict = [
                {'img': os.path.join(args.root_path, img), 'mask': os.path.join(args.root_path, mask)} for
                img, mask in zip(unlabeled_img_2, unlabeled_mask_2)]
            train_dataset = CacheDataset(data=unlabeled_dict, transform=train_transforms, cache_rate=1.0, num_workers=0)

        dict_users.append(train_dataset)
        dict_length.append(len(train_dataset))

    net_glob = BaselineUNet(in_channels=1, n_cls=num_classes, n_filters=24)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        net_glob = torch.nn.DataParallel(net_glob, device_ids=device_ids).to(device)

    net_glob.train()
    w_glob = net_glob.state_dict()
    w_locals = []
    trainer_locals = []
    net_locals = []
    optim_locals = []

    for i in supervised_user_id:
        trainer_locals.append(SupervisedLocalUpdate(args, dict_users[i], dict_length[i]))
        w_locals.append(copy.deepcopy(w_glob))
        net_locals.append(copy.deepcopy(net_glob).to(device))
        optimizer = torch.optim.AdamW(net_locals[i].parameters(), lr=args.base_lr, weight_decay=1e-5)
        optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    for i in unsupervised_user_id:
        trainer_locals.append(UnsupervisedLocalUpdate(args, dict_users[i], dict_length[i]))

    for com_round in range(1, args.rounds + 1):
        print("\nbegin com")
        loss_locals = []

        for idx in supervised_user_id:
            if com_round*args.local_ep > begin_unsupervised:
                trainer_locals[idx].base_lr = 2e-4

            local = trainer_locals[idx]
            optimizer = optim_locals[idx]
            w, loss, op = local.train(args, net_locals[idx], optimizer, com_round*args.local_ep, center_avg, c_t_avg)
            w_locals[idx] = copy.deepcopy(w)
            optim_locals[idx] = copy.deepcopy(op)
            loss_locals.append(copy.deepcopy(loss))

        if com_round*args.local_ep > begin_unsupervised:
            if not flag_create:
                print('begin unsup')
                for i in unsupervised_user_id:
                    w_locals.append(copy.deepcopy(w_glob))
                    net_locals.append(copy.deepcopy(net_glob).to(device))
                    optimizer = torch.optim.AdamW(net_locals[i].parameters(), lr=args.base_lr, weight_decay=1e-5)
                    optim_locals.append(copy.deepcopy(optimizer.state_dict()))
                flag_create = True

            for idx in unsupervised_user_id:
                local = trainer_locals[idx]
                optimizer = optim_locals[idx]
                w, loss, op = local.train(args, net_locals[idx], optimizer, com_round*args.local_ep, center_avg, c_t_avg, mean_avg, var_avg)
                w_locals[idx] = copy.deepcopy(w)
                optim_locals[idx] = copy.deepcopy(op)
                loss_locals.append(copy.deepcopy(loss))

        with torch.no_grad():
            center_sum, c_t_sum = trainer_locals[0].center, trainer_locals[0].c_t
            if com_round*args.local_ep > begin_unsupervised:

                for idx in supervised_user_id[1:]:
                    center_sum = center_sum + trainer_locals[idx].center
                    c_t_sum = c_t_sum + trainer_locals[idx].c_t

                for idx in unsupervised_user_id:
                    center_sum = center_sum + trainer_locals[idx].center
                    c_t_sum = c_t_sum + trainer_locals[idx].c_t
                center_avg = center_sum / (len(supervised_user_id) + len(unsupervised_user_id))
                c_t_avg = c_t_sum / (len(supervised_user_id) + len(unsupervised_user_id))
            else:
                for idx in supervised_user_id[1:]:
                    center_sum = center_sum + trainer_locals[idx].center
                    c_t_sum = c_t_sum + trainer_locals[idx].c_t
                center_avg = center_sum / len(supervised_user_id)
                c_t_avg = c_t_sum / len(supervised_user_id)

        with torch.no_grad():
            mean_sum, var_sum = trainer_locals[0].softmatch.mean, trainer_locals[0].softmatch.var
            for idx in supervised_user_id[1:]:
                mean_sum = mean_sum + trainer_locals[idx].softmatch.mean
                var_sum = var_sum + trainer_locals[idx].softmatch.var
            mean_avg = mean_sum / len(supervised_user_id)
            var_avg = var_sum / len(supervised_user_id)

        with torch.no_grad():
            if com_round*args.local_ep > begin_unsupervised:
                w_glob = aggregation(w_locals, dict_length, trainer_locals)
            else:
                w_glob = FedAvg(w_locals, dict_length)
        net_glob.load_state_dict(w_glob)


        for i in supervised_user_id:
            net_locals[i].load_state_dict(w_glob)

        if com_round*args.local_ep > begin_unsupervised:
            for i in unsupervised_user_id:
                net_locals[i].load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        # print(loss_avg, com_round)
        logging.info('Loss Avg {}, Round {}, LR {} '.format(loss_avg, com_round, args.base_lr))

        if com_round % 50 == 0:
            logging.info("TEST: Epoch: {}".format(com_round))

            save_global_path = os.path.join(snapshot_path, 'global_epoch_' + str(com_round) + '.pth')
            # torch.save({'state_dict': net_glob.module.state_dict(), }, save_mode_path)
            torch.save({'state_dict': w_glob, }, save_global_path)

            dice_avg_val, hd_avg_val, asd_avg_val, nsd_avg_val, jc_avg_val = Validate(save_global_path, com_round)
            dice_avg, hd_avg, asd_avg, nsd_avg, jc_avg = Test(save_global_path, com_round)



