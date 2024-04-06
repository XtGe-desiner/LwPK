import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from dataloader.miniimagenet.autoaugment import AutoAugImageNetPolicy


class MiniImageNet(Dataset):

    def __init__(self, root='/data/', train=True,
                 transform=None,
                 index_path=None, index=None, new_index=None, base_sess=None, autoaug=1, supervised=True):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.IMAGE_PATH = root + 'miniimagenet/images'
        self.SPLIT_PATH = root + 'index_list/mini_imagenet'
        csv_path = self.SPLIT_PATH + '/' + setname + '.csv'
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            # path = osp.join(self.IMAGE_PATH, name)
            path = self.IMAGE_PATH + '/' + name
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb

        if autoaug == 0:
            # do not use autoaug.
            if train:
                image_size = 84
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
                if base_sess:
                    if supervised:
                        self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                    else:
                        self.data, self.targets = self.SelectFromClasses2(self.data, self.targets, new_index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            else:
                image_size = 84
                self.transform = transforms.Compose([
                    transforms.Resize([92, 92]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        else:
            # use autoaug.
            if train:
                image_size = 84
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    # add autoaug
                    AutoAugImageNetPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                if base_sess:
                    if supervised:
                        self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                    else:
                        self.data, self.targets = self.SelectFromClasses2(self.data, self.targets, new_index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            else:
                image_size = 84
                self.transform = transforms.Compose([
                    transforms.Resize([92, 92]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def SelectfromTxt(self, data2label, index_path):
        # select from txt file, and make cooresponding mampping.
        index = []
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            # img_path = os.path.join(self.IMAGE_PATH, i)
            img_path = self.IMAGE_PATH + '/' + i
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectNotFromTxt(self):
        image_list = []
        sessions = 9
        for session in range(1, sessions):
            txt_path = "./data/index_list/mini_imagenet/session_" + str(session + 1) + '.txt'
            all_txt = open(txt_path).read().splitlines()
            for i in all_txt:
                image_list.append(i)
        for i in image_list:
            i = './data/miniimagenet/images/' + i[-21:]
            for k in list(self.data2label.keys()):
                v = self.data2label[k]
                if i == k:
                    self.data2label.pop(k)
                    self.data.remove(k)
                    self.targets.remove(v)
        return self.data, self.targets

    def SelectfromClasses(self, data, targets, index):
        # select from csv file, choose all instances from this class.
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def SelectFromClasses2(self, data, targets, new_index):
        # data_tmp, target_tmp = self.SelectfromClasses(self.data, self.targets, index)
        data, target = self.SelectNotFromTxt()
        data_tmp = []
        target_tmp = []
        for i in new_index:
            ind_cl1 = np.where(i == target)[0]
            # randnum = random.randint(1, 3)
            randlist = random.sample(list(ind_cl1), 50)
            for j in randlist:
                data_tmp.append(data[j])
                target_tmp.append(target[j])
        return data_tmp, target_tmp

    def SelectUnsupervisedData(self, new_index, randnum):
        data_tmp = []
        data, target = self.SelectNotFromTxt()
        for i in new_index:
            ind_cl1 = np.where(i == target)[0]
            randlist = random.sample(range(ind_cl1[0], ind_cl1[-1] + 1), randnum)
            for j in randlist:
                data_tmp.append(data[j])
                random.shuffle(data_tmp)
        return data_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets

