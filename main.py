#! /usr/bin/env python
import os
import argparse
import pickle
import random
import sys
from typing import Tuple, NoReturn

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import tqdm.autonotebook as tqdm
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models
from torchvision.models import resnet
from dataloader.medicalmnist.medicalmnist import MedicalMnist


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", type=str, default="0")
    parser.add_argument("-n", "--num_workers", type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./runs', help='path for saving')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['miniImageNet', 'cub200', 'cifar100'],
                        help='dataset')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    return args


def main():
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = [transforms.RandomResizedCrop(size=32,
                                       scale=(0.2, 1.0),
                                       ratio=(3 / 4, 4 / 3)),
          transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
          transforms.RandomGrayscale(p=0.2),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
          ]
    transform = transforms.Compose(tf)
    if args.dataset == 'cifar100':
        trainset = CIFAR100(root="./datasets",
                            train=True,
                            transform=transform)


    elif args.dataset == 'cub200':
        tf = [transforms.RandomResizedCrop(size=224,
                                           scale=(0.2, 1.0),
                                           ratio=(3 / 4, 4 / 3)),
              transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
              transforms.RandomGrayscale(p=0.2),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
              ]
        transform = transforms.Compose(tf)
        trainset = CUB200(root="./datasets",
                          train=True,
                          transform=transform)
    elif args.dataset == 'miniImageNet':
        trainset = MiniImageNet(root="./datasets/",
                                train=True,
                                transform=transform)
    else:
        raise ValueError('No such data')

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=128,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers)

    low_dim = 128
    if args.dataset == 'cub200':
        net = load_pretrained_resNet18(low_dim)
    else:
        net = ResNet18(low_dim=low_dim)

    norm = Normalize(2)
    npc = NonParametricClassifier(input_dim=low_dim,
                                  output_dim=len(trainset),
                                  tau=1.0,
                                  momentum=0.5)
    loss = Loss(tau2=2.0)
    net, norm = net.to(device), norm.to(device)
    npc, loss = npc.to(device), loss.to(device)
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=False,
                                dampening=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        [50, 100, 150, 200],
                                                        gamma=0.25)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=range(len(
            args.gpus.split(","))))
        torch.backends.cudnn.benchmark = True

    trackers = {n: AverageTracker() for n in ["loss", "loss_id", "loss_fd"]}
    with tqdm.trange(args.epochs) as epoch_bar:
        for epoch in epoch_bar:
            net.train()
            for batch_idx, (inputs, _, indexes) in enumerate(tqdm.tqdm(train_loader)):
                optimizer.zero_grad()
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                indexes = indexes.to(device, non_blocking=True)
                features = norm(net(inputs))
                outputs = npc(features, indexes)
                loss_id, loss_fd = loss(outputs, features, indexes)
                tot_loss = loss_id + loss_fd
                tot_loss.backward()
                optimizer.step()
                # track loss
                trackers["loss"].add(tot_loss)
                trackers["loss_id"].add(loss_id)
                trackers["loss_fd"].add(loss_fd)
            lr_scheduler.step()

            # logging
            postfix = {name: t.avg() for name, t in trackers.items()}
            epoch_bar.set_postfix(**postfix)
            for t in trackers.values():
                t.reset()

            # check clustering acc
            if (epoch == 0) or (((epoch + 1) % 10) == 0):
                acc, nmi, ari = check_clustering_metrics(npc, train_loader)
                print("Epoch:{} Kmeans ACC, NMI, ARI = {}, {}, {}".format(epoch + 1, acc, nmi, ari))

    args.save_path = args.save_path + '/' + args.dataset
    if (args.save_path is not None) and (not os.path.isdir(args.save_path)):
        os.makedirs(args.save_path)

    if args.save_path is not None:
        save_file = os.path.join(args.save_path, 'last.pth')
        save_model(net, args.epochs, save_file)
        # save_npc = os.path.join(args.save_path, 'npc_last.pth')
        # save_file()


class AverageTracker():
    def __init__(self):
        self.step = 0
        self.cur_avg = 0

    def add(self, value):
        self.cur_avg *= self.step / (self.step + 1)
        self.cur_avg += value / (self.step + 1)
        self.step += 1

    def reset(self):
        self.step = 0
        self.cur_avg = 0

    def avg(self):
        return self.cur_avg.item()


class CIFAR100(Dataset):
    """CIFAR20 Dataset.

    This is a subclass of the `CIFAR100` Dataset.
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

    def __init__(self, root='./dataset/',
                 train=True,
                 download: bool = False,
                 transform=None):

        super(CIFAR100, self).__init__()
        self.root = root
        self.transform = transform

        self.transform1 = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        self.train = train  # train or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.labels = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels.extend(entry['labels'])
                else:
                    self.labels.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.labels = np.asarray(self.labels)

        index = np.arange(60, 100)
        self.data, self.labels = self.SelectfromDefault(self.data, self.labels, index)

        self._load_meta()

    def SelectNotFromTxt(self):
        index_list = []
        sessions = 9
        for session in range(1, sessions):
            txt_path = "./data/index_list/cifar100/session_" + str(
                session + 1) + '.txt'
            all_txt = open(txt_path).read().splitlines()
            for i in all_txt:
                index_list.append(int(i))

        return index_list

    def SelectfromDefault(self, data, targets, index):
        index_list = self.SelectNotFromTxt()
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            ind_cl = random.sample(list(ind_cl), 100)
            ind_cl = sorted(list(set(ind_cl).difference(set(index_list))))
            if data_tmp == []:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

        return data_tmp, targets_tmp

    def _load_meta(self) -> NoReturn:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index
        Returns:
            Tuple: image, class
        """
        img, target = Image.fromarray(self.data[index]), self.labels[index]
        if self.transform is None:
            img = self.transform1(img)
        else:
            img = self.transform(img)

        return img, target, index

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> NoReturn:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)


class CUB200(Dataset):

    def __init__(self, root='./data/',
                 train=False,
                 transform=None):
        self.root = os.path.expanduser(root)
        self.train = train  # train or test set
        self._pre_operate(self.root)
        self.transform = transform

        index = np.arange(100, 200)
        self.data, self.labels = self.SelectfromClasses(self.data, self.labels, index)

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
        self.labels = []
        self.data2label = {}
        index_list = self.SelectNotFromTxt()
        train_idx = list(set(train_idx).difference(set(index_list)))
        if self.train:
            for k in train_idx:
                image_path = root + '/CUB_200_2011/images/' + id2image[k]
                self.data.append(image_path)
                self.labels.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.labels.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectNotFromTxt(self):
        index_list = []
        image_list = []
        sessions = 11
        for session in range(1, sessions):
            txt_path = "./data/index_list/cub200/session_" + str(session + 1) + '.txt'
            all_txt = open(txt_path).read().splitlines()
            for i in all_txt:
                image_list.append(i)
        for i in image_list:
            i = i.replace('CUB_200_2011/images/', './data/CUB_200_2011/images/')
            if i in self.data:
                index_list.append(self.data.index(i))
        return index_list

    def SelectfromClasses(self, data, labels, index):
        index_list = self.SelectNotFromTxt()
        data_tmp = []
        labels_tmp = []
        for i in index:
            ind_cl = np.where(i == labels)[0]
            ind_cl = list(set(ind_cl) - set(index_list))
            for j in ind_cl:
                data_tmp.append(data[j])
                labels_tmp.append(labels[j])
        return data_tmp, labels_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, labels = self.data[i], self.labels[i]
        image = self.transform(Image.open(path).convert('RGB'))

        return image, labels, i


class MiniImageNet(Dataset):

    def __init__(self, root='./data/',
                 train=True,
                 transform=None):

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
        self.labels = []
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
            self.labels.append(lb)
            self.data2label[path] = lb

            index = np.arange(60, 100)
            self.data, self.labels = self.SelectfromClasses(self.data, self.labels, index)

    def SelectNotFromTxt(self):
        image_list = []
        sessions = 9
        for session in range(1, sessions):
            txt_path = "./data/index_list/mini_imagenet/session_" + str(session + 1) + '.txt'
            all_txt = open(txt_path).read().splitlines()
            for i in all_txt:
                image_list.append(i)
        for i in image_list:
            i = './data/index_list/miniimagenet/images/' + i[-21:]
            for k in list(self.data2label.keys()):
                v = self.data2label[k]
                if i == k:
                    self.data2label.pop(k)
                    self.data.remove(k)
                    self.labels.remove(v)
        return self.data, self.labels

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.labels[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets, i


def check_clustering_metrics(npc, train_loader):
    trainFeatures = npc.memory
    z = trainFeatures.cpu().numpy()
    y = np.array(train_loader.dataset.targets)
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z)
    return metrics.acc(y, y_pred), metrics.nmi(y,
                                               y_pred), metrics.ari(y, y_pred)


def save_model(model, epoch, save_file):
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


class metrics:
    ari = adjusted_rand_score
    nmi = normalized_mutual_info_score

    @staticmethod
    def acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row, col = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(row, col)]) * 1.0 / y_pred.size


class NonParametricClassifierOP(Function):
    @staticmethod
    def forward(ctx, x, y, memory, params):
        tau = params[0].item()
        out = x.mm(memory.t())
        out.div_(tau)
        ctx.save_for_backward(x, memory, y, params)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, memory, y, params = ctx.saved_tensors
        tau = params[0]
        momentum = params[1]

        grad_output.div_(tau)

        grad_input = grad_output.mm(memory)
        grad_input.resize_as_(x)

        weight_pos = memory.index_select(0, y.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(x.mul(1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None, None


class NonParametricClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, tau=1.0, momentum=0.5):
        super(NonParametricClassifier, self).__init__()
        self.register_buffer('params', torch.tensor([tau, momentum]))
        stdv = 1. / np.sqrt(input_dim / 3.)
        self.register_buffer(
            'memory',
            torch.rand(output_dim, input_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = NonParametricClassifierOP.apply(x, y, self.memory, self.params)
        return out


class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def ResNet18(low_dim=128):
    net = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], low_dim)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                          stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


def load_pretrained_resNet18(low_dim=128):
    net = models.resnet18(pretrained=True)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                          stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    final_inplanes = net.fc.in_features
    net.fc = nn.Linear(final_inplanes, low_dim)
    return net


class Loss(nn.Module):
    def __init__(self, tau2):
        super().__init__()
        self.tau2 = tau2

    def forward(self, x, ff, y):
        L_id = F.cross_entropy(x, y)

        norm_ff = ff / (ff ** 2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_ff.t(), norm_ff)
        coef_mat.div_(self.tau2)
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)
        L_fd = F.cross_entropy(coef_mat, a)
        return L_id, L_fd


if __name__ == "__main__":
    main()
