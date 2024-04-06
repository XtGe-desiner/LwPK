# import new Network name here and add in model_class args
import numpy as np
import torch.utils.data
from torch.autograd import Function
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch.nn as nn
import torchvision.models as models
from numpy import linalg
from sklearn.utils import *
from collections import Counter
from imutils import build_montages
from models.resnet18_encoder import *
from sklearn import decomposition


def data_concate(train_set, unsuperset, args):
    if args.dataset == 'cifar100':
        train_set.data = np.vstack((train_set.data, unsuperset.data))
        train_set.targets = np.hstack((train_set.targets, unsuperset.targets))
    else:
        for i in range(len(unsuperset.data)):
            train_set.data.append(unsuperset.data[i])
            train_set.targets.append(unsuperset.targets[i])
    return train_set


def pub_pseudo_label(unsuperset, train_set, args):
    unsuperload = torch.utils.data.DataLoader(dataset=unsuperset, batch_size=1, num_workers=0, shuffle=False,
                                              pin_memory=True)
    unsuperlist = []
    NET = resnet18(pretrained=True, progress=True)
    net = NET.cuda()
    net.eval()
    for i, batch1 in enumerate(unsuperload, 0):
        unsuperdata = [_.cuda() for _ in batch1]
        unsuperimage = net(unsuperdata[0])
        unsuperimage = unsuperimage.reshape(-1, )
        unsuperlist.append(unsuperimage.detach().cpu().numpy())

    clt = KMeans(n_clusters=args.num_classes - args.base_class)
    clt.fit(unsuperlist)
    clt.labels_ = clt.labels_ + args.base_class
    if args.dataset == 'cifar100':
        train_set.data = np.vstack((train_set.data, unsuperset.data))
        train_set.targets = np.hstack((train_set.targets, unsuperset.targets))
    else:
        for i in range(len(unsuperset.data)):
            train_set.data.append(unsuperset.data[i])
            train_set.targets.append(clt.labels_[i])
    ari = adjusted_rand_score(clt.labels_, unsuperset.targets)
    print("kmeans ari: ", ari)
    return train_set


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()

    #     standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        logits = model(data)
        # logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label.long())
        acc = count_acc(logits, train_label)
        total_loss = loss
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=0, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label.long())
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va, va5))

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            print('Seen Acc:', seenac, 'Unseen ACC:', unseenac)
    return vl, va