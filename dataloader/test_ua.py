import torch
import torch.nn.functional as F
import numpy as np
from numpy import linalg
import random
import torchvision.models as models
import copy
import torch.nn as nn
from dataloader.cifar100.cifar import CIFAR100
from dataloader.cub200.cub200 import CUB200
from dataloader.miniimagenet.miniimagenet import MiniImageNet
from train import get_command_line_parser
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def choose_data(unlabeled_dataset, ref_model, args):
    data_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=16, shuffle=True, num_workers=0)
    u_t = args.UA
    high_ua_dataset = copy.deepcopy(unlabeled_dataset)
    ref_model.eval()
    idx_del = []
    select_indexes = []
    for idx, batch in enumerate(data_loader):
        inputs, label, select_index = [_ for _ in batch]
        out_prob = []
        select_index = select_index.tolist()
        select_indexes.extend(select_index)
        for _ in range(10):
            noise = torch.clamp(torch.randn_like(inputs) * 0.01, -0.02, 0.02)
            inputs_noise = inputs + noise
            outputs_noise = ref_model(inputs_noise)
            out_prob.append(F.softmax(outputs_noise, dim=1))

        out_prob = torch.stack(out_prob)
        out_std = torch.std(out_prob, dim=0)
        out_prob = torch.mean(out_prob, dim=0)
        max_value, max_idx = torch.max(out_prob, dim=1)
        max_std = out_std.gather(1, max_idx.view(-1, 1))
        max_std_sorted, std_indices = torch.sort(max_std, 0, descending=False)
        max_std = max_std.squeeze(1).detach().cpu().numpy()

        for idx in range(len(max_std)):
            if max_std[idx] >= max_std_sorted[int(u_t * len(max_std))]:
                idx_del.append(select_index[idx])
    final_index = list(set(select_indexes) - set(idx_del))
    select_data = [unlabeled_dataset.data[i] for i in final_index]
    select_label = [unlabeled_dataset.targets[i] for i in final_index]

    high_ua_data = [unlabeled_dataset.data[i] for i in idx_del]
    high_ua_label = [unlabeled_dataset.targets[i] for i in idx_del]

    unlabeled_dataset.data = select_data
    unlabeled_dataset.targets = select_label

    high_ua_dataset.data = high_ua_data
    high_ua_dataset.targets = high_ua_label
    high_ua_dataset.need_index = False

    return unlabeled_dataset, high_ua_dataset


def pseudo_label(extra_dataset, high_ua_dataset, model, args):
    unSuperLoad = torch.utils.data.DataLoader(dataset=extra_dataset, batch_size=1, num_workers=0,
                                              shuffle=False, pin_memory=True)

    unSuperList = []
    target_list = []
    high_ua_List = []
    last_targets = []
    high_ua_gt_list = []

    model.eval()
    for i, batch1 in enumerate(unSuperLoad, 0):
        unSuperData, targets, _ = [_ for _ in batch1]
        target_list.append(targets.item())
        unSuperImage = model(unSuperData)
        unSuperImage = unSuperImage.reshape(-1, )
        unSuperList.append(unSuperImage.detach().cpu().numpy())

    unSuperList = np.array(unSuperList)
    n_cluster = args.num_classes - args.base_class

    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    predict_label = kmeans.fit_predict(unSuperList) + args.base_class
    extra_dataset.targets = predict_label
    ARS = adjusted_rand_score(target_list, predict_label)
    print("ARS:{}".format(ARS))

    return extra_dataset


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def CIFAR_PL_dataset(args):
    set_seed(8)
    ref_model = models.resnet50(pretrained=True)
    new_class_index = np.arange(60, 100)
    unlabeled_dataset = CIFAR100('./data/', train=True, download=False, new_index=new_class_index,
                                 base_sess=True, autoaug=0, supervised=False, need_index=True)
    unlabeled_dataset, high_ua_dataset = choose_data(unlabeled_dataset, ref_model, args)
    pseudo_label_dataset = pseudo_label(unlabeled_dataset, high_ua_dataset, ref_model, args)
    return pseudo_label_dataset


def CUB_PL_dataset(args, i):
    set_seed(i)
    ref_model = models.resnet18(pretrained=True)
    new_class_index = np.arange(100, 200)
    unlabeled_dataset = CUB200('./data/', train=True, new_index=new_class_index, base_sess=True, autoaug=0,
                               supervised=False, need_index=True)
    unlabeled_dataset, high_ua_dataset = choose_data(unlabeled_dataset, ref_model, args)
    pseudo_label_dataset = pseudo_label(unlabeled_dataset, high_ua_dataset, ref_model, args)
    return pseudo_label_dataset


def MI_PL_dataset(args):
    ref_model = models.resnet50(pretrained=True)
    ref_model.cuda()
    data_index = True
    new_class_index = np.arange(60, 100)
    unlabeled_dataset = MiniImageNet('./data/', train=True, new_index=new_class_index, base_sess=True, autoaug=0,
                                     supervised=False, need_index=data_index)
    unlabeled_dataset, high_ua_dataset = choose_data(unlabeled_dataset, ref_model, args)
    pseudo_label_dataset = pseudo_label(unlabeled_dataset, high_ua_dataset, ref_model, args)
    return pseudo_label_dataset
