import numpy as np
import torch
from dataloader.test_ua import CIFAR_PL_dataset, CUB_PL_dataset, MI_PL_dataset
from evaluate import pseudo_label


def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11

    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    args.Dataset = Dataset
    return args


def get_base_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    class_index1 = np.arange(args.base_class, args.num_classes)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True, index=class_index,
                                         new_index=class_index1, base_sess=True)

        if args.use_UA:
            unsuperset = CIFAR_PL_dataset(args)
        else:
            unsuperset = args.Dataset.CIFAR100(root=args.dataroot, autoaug=0, train=True, download=True,
                                               new_index=class_index1, base_sess=True, supervised=False, need_index=True)
            if args.use_pub:
                pass
            else:
                unsuperset = pseudo_label(unsuperset)

        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index=class_index, base_sess=True)
        if args.use_UA:
            unsuperset = CUB_PL_dataset(args)
        else:
            unsuperset = args.Dataset.CUB200(root=args.dataroot, autoaug=0, train=True, new_index=class_index1,
                                             base_sess=True, supervised=False)
            if args.use_pub:
                pass
            else:
                unsuperset = pseudo_label(unsuperset)

        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        if args.use_UA:
            unsuperset = MI_PL_dataset(args)
        else:
            unsuperset = args.Dataset.MiniImageNet(root=args.dataroot, autoaug=0, train=True, new_index=class_index1,
                                                   base_sess=True, supervised=False)
            if args.use_pub:
                pass
            else:
                unsuperset = pseudo_label(unsuperset)

        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=0, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader( dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True)
    return trainset, unsuperset, trainloader, testloader


def get_new_dataloader(args, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    class_index1 = np.arange(args.base_class + (session-1) * args.way, args.base_class + session * args.way)
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
        unsuperset = testset
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
        unsuperset = testset
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_new)
        unsuperset = testset

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, unsuperset, trainloader, testloader


def get_session_classes(args, session):
    class_list = np.arange(args.base_class + session * args.way)
    return class_list
