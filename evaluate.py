from main import *


def pseudo_label(trainset):
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
    best_model_dict = torch.load('D:/CVPR22-Fact-main/runs/mnist/last.pth')[
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
            # indexes = indexes.to(device, non_blocking=True)
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
    trainset.targets = y_pred + 35
    return trainset

# def pseudo_label(trainset):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train_loader = torch.utils.data.DataLoader(trainset,
#                                                batch_size=128,
#                                                shuffle=False,
#                                                pin_memory=True,
#                                                num_workers=0)
#     low_dim = 128
#     net = load_pretrained_resNet18(low_dim)
#     net = torch.nn.DataParallel(net)
#     best_model_dict = torch.load('D:/CVPR22-Fact-main/runs/miniimagenet/last.pth')[
#         'model']
#     net.load_state_dict(best_model_dict)
#
#     norm = Normalize(2)
#     net, norm = net.to(device), norm.to(device)
#     net.eval()
#     out_std_list = []
#     with torch.no_grad():
#         for batch_idx, (inputs, _) in enumerate(tqdm.tqdm(train_loader)):
#             out_prob = []
#             for _ in range(10):
#                 noise = torch.clamp(torch.randn_like(inputs) * 0.01, -0.02, 0.02)
#                 inputs_noise = inputs + noise
#                 outputs_noise = net(inputs_noise)
#                 # out_prob.append(F.softmax(outputs_noise, dim=1))
#                 out_prob.append(outputs_noise)
#
#             out_prob = torch.stack(out_prob)
#             out_std = torch.std(out_prob, dim=0)
#             out_prob = torch.mean(out_prob, dim=0)
#
#             max_value, max_idx = torch.max(out_prob, dim=1)
#             max_std = out_std.gather(1, max_idx.view(-1, 1))
#
#             max_std = max_std.squeeze(1).detach().cpu().numpy()
#             out_std_list.extend(max_std.tolist())
#
#         out_std_list = np.array(out_std_list)
#
#     features_list = None
#     with torch.no_grad():
#         for batch_idx, (inputs, _) in enumerate(tqdm.tqdm(train_loader)):
#             inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
#             features = norm(net(inputs)).detach().cpu().numpy()
#             if features_list is None:
#                 features_list = features
#             else:
#                 features_list = np.concatenate([features_list, features], axis=0)
#
#     y = np.array(train_loader.dataset.targets)
#     unique_label = np.unique(y)
#     n_clusters = len(unique_label)
#     kmeans = KMeans(n_clusters=n_clusters, n_init=20)
#     y_pred = kmeans.fit_predict(features_list)
#     ari = metrics.ari(y, y_pred)
#     print("Kmeans ARI = {}".format(ari))
#
#     trainset.targets = y_pred
#     top_50_indices = []
#     for i in range(n_clusters):
#         id = np.where(trainset.targets == i)[0]
#         select_std_list = out_std_list[id]
#         sorted_indices = np.argsort(select_std_list)[::-1]
#         select_id_len = 50 if len(id) >= 50 else len(id)
#         if top_50_indices == []:
#             top_50_indices = id[sorted_indices[:select_id_len]]
#         else:
#             top_50_indices = np.hstack((top_50_indices, id[sorted_indices[:select_id_len]]))
#     if n_clusters == 40:
#         plus_class = 60
#     else:
#         plus_class = 100
#
#     final_data = [trainset.data[i] for i in top_50_indices]
#     # final_targets = [trainset.targets[i] for i in top_50_indices]
#     final_targets = trainset.targets[top_50_indices]
#     trainset.data = final_data
#     trainset.targets = final_targets + plus_class
#
#     final_ari = metrics.ari(y[top_50_indices], final_targets)
#     print("Kmeans Final ARI = {}".format(final_ari))
#
#     return trainset


