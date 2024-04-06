import numpy as np

from main import *


def pseudo_label(trainset):
    if len(np.unique(trainset.targets)) == 100:
        plus_num = 100
    else:
        plus_num = 60

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=128,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=0)
    low_dim = 128
    # net = load_pretrained_resNet18(low_dim)
    net = ResNet18(low_dim)
    net = torch.nn.DataParallel(net)
    best_model_dict = torch.load('./runs/____/last.pth')[  # due to the actual .pth file path
        'model']
    net.load_state_dict(best_model_dict)
    norm = Normalize(2)
    npc = NonParametricClassifier(input_dim=low_dim,
                                  output_dim=len(trainset),
                                  tau=1.0,
                                  momentum=0.5)
    net, norm, npc = net.to(device), norm.to(device), npc.to(device)
    net.eval()
    npc.eval()
    features_list = None
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm.tqdm(train_loader)):
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            features = norm(net(inputs)).detach().cpu().numpy()
            if features_list is None:
                features_list = features
            else:
                features_list = np.concatenate([features_list, features], axis=0)

    y = np.array(train_loader.dataset.targets)
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(features_list)
    ari = metrics.ari(y, y_pred)
    print("Kmeans ARI = {}".format(ari))
    trainset.targets = y_pred + plus_num
    return trainset


