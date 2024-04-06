import random

import torch
from PIL import Image
import os
import os.path
import numpy as np
import pickle

import torchvision.transforms as transforms

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from dataloader.cifar100.autoaugment import CIFAR10Policy, Cutout


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, index=None, new_index=None, base_sess=None, autoaug=1, supervised=True, need_index=False):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.need_index = need_index

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if autoaug==0:
            if self.train:
                downloaded_list = self.train_list
                self.transform = transforms.Compose([
                    transforms.Resize([32, 32]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            else:
                downloaded_list = self.test_list
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ])
        else:
            if self.train:
                downloaded_list = self.train_list
                self.transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),    # add AutoAug
                    transforms.ToTensor(),
                    Cutout(n_holes=1, length=16),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
            else:
                downloaded_list = self.test_list
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])

        if autoaug == 0 and supervised == False:
            self.transform = transforms.Compose([transforms.RandomResizedCrop(size=32,
                                          scale=(0.2, 1.0),
                                          ratio=(3 / 4, 4 / 3)),
             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
             transforms.RandomGrayscale(p=0.2),
             transforms.ToTensor(),
             transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
             ])

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = np.asarray(self.targets)
        if base_sess:
            if train:
                if supervised:
                     self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)
                else:
                     self.data, self.targets = self.SelectfromDefault2(self.data, self.targets, new_index)
            else:
                self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)
        else:  # new Class session
            if train:
                self.data, self.targets = self.NewClassSelector(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)

        self._load_meta()

    def SelectfromDefault(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            if data_tmp == []:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

        return data_tmp, targets_tmp

    def SelectNotFromTxt(self):
        index_list = []
        sessions = 9
        for session in range(1,sessions):
            txt_path = "./data/index_list/cifar100/session_" + str(session + 1) + '.txt'
            all_txt = open(txt_path).read().splitlines()
            for i in all_txt:
                index_list.append(int(i))
        return index_list

    def SelectfromDefault2(self, data, targets, new_index):  # 加载无标签数据
        index_list = self.SelectNotFromTxt()
        data_tmp = []
        targets_tmp = []
        for i in new_index:
            ind_cl1 = np.where(i == targets)[0]
            ind_cl2 = sorted(list(set(ind_cl1).difference(set(index_list))))  # 确保后期出现的这里不会出现
            # randnum = random.randint(30, 50)
            # randlist = random.sample(list(ind_cl2), randnum)
            for j in ind_cl2:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, targets_tmp

    def SelectfromDefault3(self, data, targets, new_index):  # 加载所有数据，一个gt
        index = np.arange(60)
        data_tmp, targets_tmp = self.SelectfromDefault(self.data, self.targets, index)

        data_tmp = data_tmp.tolist()
        targets_tmp = targets_tmp.tolist()

        index_list = self.SelectNotFromTxt()
        # pl_list = np.arange(60, 100)
        # pl_index = 0
        for i in new_index:
            ind_cl1 = np.where(i == targets)[0]
            ind_cl2 = sorted(list(set(ind_cl1).difference(set(index_list))))  # 确保后期出现的这里不会出现
            randlist = random.sample(list(ind_cl2), 100)
            for j in randlist:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        data_tmp = np.array(data_tmp)
        targets_tmp = np.array(targets_tmp)
        return data_tmp, targets_tmp

    def NewClassSelector(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        ind_list = [int(i) for i in index]
        ind_np = np.array(ind_list)
        index = ind_np.reshape((5,5))
        for i in index:
            ind_cl = i
            if data_tmp == []:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

        return data_tmp, targets_tmp

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(np.uint8(img))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.need_index:
            return img, target, index

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
