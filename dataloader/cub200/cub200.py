import os
import os.path as osp
import numpy as np
import random
import torch
from utils import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataloader.cub200.autoaugment import AutoAugImageNetPolicy


class CUB200(Dataset):

    def __init__(self, root='/data/', train=True,
                 index_path=None, index=None, new_index=None, base_sess=None, autoaug=1, supervised=True,
                 need_index=False):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self._pre_operate(self.root)
        self.need_index = need_index

        if autoaug == 0:
            # do not use autoaug
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                if base_sess:
                    if supervised:
                        self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                    else:
                        self.data, self.targets = self.SelectUnsupervisedData(self.data, self.targets, new_index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        else:
            # use autoaug
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    # add autoaug
                    AutoAugImageNetPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                if base_sess:
                    if supervised:
                        self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                    else:
                        self.data = self.SelectUnsupervisedData(self.data, self.targets, index, new_index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        # image_file = root +'/'+ 'CUB_200_2011/images.txt'
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        # index_list = self.SelectNotFromTxt()
        # train_idx = list(set(train_idx).difference(set(index_list)))
        if self.train:
            for k in train_idx:
                image_path = root + '/CUB_200_2011/images/' + id2image[k]
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = self.root + '/' + i
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectNotFromTxt(self):
        index_list = []
        image_list = []
        sessions = 11
        image_file = './data/CUB_200_2011/images.txt'
        id2image = self.list2dict(self.text_read(image_file))
        for session in range(1, sessions):
            txt_path = "./data/index_list/cub200/session_" + str(session + 1) + '.txt'
            all_txt = open(txt_path).read().splitlines()
            for i in all_txt:
                image_list.append(i)
        for i in image_list:
            i = i.replace('CUB_200_2011/images/', '')
            for k, v in id2image.items():
                if i == v:
                    index_list.append(k)
        return index_list

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, targets_tmp

    def SelectfromClasses2(self, data, targets, index, new_index):
        data_tmp, targets_tmp = self.SelectfromClasses(self.data, self.targets, index)
        for i in new_index:
            ind_cl1 = np.where(i == targets)[0]
            length1 = len(ind_cl1)
            randnum = random.randint(1, 3)
            randlist = random.sample(range(ind_cl1[0], ind_cl1[-1] + 1), randnum)
            for j in randlist:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, targets_tmp

    def SelectUnsupervisedData(self, data, targets, new_index):
        data_tmp = []
        target_tmp = []
        for i in new_index:
            ind_cl1 = np.where(i == targets)[0]
            # randnum = random.randint(30, 50)
            # randlist = random.sample(list(ind_cl1), randnum)
            for j in ind_cl1:
                data_tmp.append(data[j])
                target_tmp.append(targets[j])
        return data_tmp, target_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))

        if self.need_index:
            return image, targets, i

        return image, targets
