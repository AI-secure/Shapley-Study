import numpy as np
import os
import os.path as osp
from sklearn.utils import shuffle

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from shapley.utils.utils import *
from shapley.use_case.plot.plot_resnet import plot_tiny_train

device = torch.device('cuda')
result_path = "embedding_result/da/"

if not osp.exists(result_path):
    os.makedirs(result_path)

tiny_path = "embedding_data/tinyimagenet/resnet18_"
data_path = "embedding_data/da/"

LOAD_ORIGINAL_DATA = True
CAL_DEEP_FEATURES = False

EMBED_SV = True
RAW_SV = False
DEEP_SV = False

print("Load original data or not:")
if LOAD_ORIGINAL_DATA == True:
    print("Load original data")

    train_num = 2500
    cal_sv_num = 2500
    pre_num = 95000

    if RAW_SV:
        # load raw data
        data_dir = "data/tiny-imagenet-200"
        data_transforms = transforms.Compose([transforms.ToTensor()])
        image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
        batch_size = len(image_datasets) # 1000 for test

        # change dataloaders to numpy
        train_dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=False, num_workers=64)
        raw_train_X, raw_train_Y = next(iter(train_dataloader))
        raw_train_X = raw_train_X.numpy() # (100000, 3, 64, 64)
        raw_train_Y = raw_train_Y.numpy() # (100000, )
        print("raw data shape: ", raw_train_X.shape, raw_train_Y.shape)

        raw_train_X, raw_train_Y = shuffle(raw_train_X, raw_train_Y, random_state=0)

        raw_ori_train_X = raw_train_X[:train_num]
        raw_ori_train_Y = raw_train_Y[:train_num]
        raw_cal_sv_X = raw_train_X[train_num:train_num + cal_sv_num]
        raw_cal_sv_Y = raw_train_Y[train_num:train_num + cal_sv_num]
        raw_pre_train_X = raw_train_X[-pre_num:]
        raw_pre_train_Y = raw_train_Y[-pre_num:]

        np.savez_compressed(data_path + "raw_train/" + "raw_ori_train.npz", x=raw_ori_train_X, y=raw_ori_train_Y)
        np.savez_compressed(data_path + "raw_train/" + "raw_cal_sv.npz", x=raw_cal_sv_X, y=raw_cal_sv_Y)

        gap = int(raw_pre_train_X.shape[0] / 10)
        print("gap: ", gap)
        for i in range(10):
            np.savez_compressed(data_path + "raw_train/" + "raw_pre_train_" + str(i) + ".npz", x=raw_pre_train_X[i * gap:i * gap + gap], y=raw_pre_train_Y[i * gap:i * gap + gap])
        print("===Data saved===")


    if EMBED_SV:
        # load embedding train data
        for i in range(100): # 100
            loader = np.load(tiny_path+str(i)+".npz")
            x = loader["x"]
            y = loader["y"]
            if i == 0:
                train_X = x
                train_Y = y
            else:
                train_X = np.concatenate((train_X, x), axis=0)
                train_Y = np.concatenate((train_Y, y), axis=0)
        print("train data shape: ", train_X.shape, train_Y.shape) # (100000, 512, 1, 1) (100000,)

        train_X, train_Y = shuffle(train_X, train_Y, random_state=0)

        ori_train_X = train_X[:train_num]
        ori_train_Y = train_Y[:train_num]
        cal_sv_X = train_X[train_num:train_num + cal_sv_num]
        cal_sv_Y = train_Y[train_num:train_num + cal_sv_num]
        pre_train_X = train_X[-pre_num:]
        pre_train_Y = train_Y[-pre_num:]

        # save data
        np.savez_compressed(data_path + "train/" + "ori_train.npz", x=ori_train_X, y=ori_train_Y)
        np.savez_compressed(data_path + "train/" + "cal_sv.npz", x=cal_sv_X, y=cal_sv_Y)

        gap = int(pre_train_X.shape[0] / 10)
        print("gap: ", gap)
        for i in range(10):
            np.savez_compressed(data_path + "train/" + "pre_train_" + str(i) + ".npz", x=pre_train_X[i * gap:i * gap + gap], y=pre_train_Y[i * gap:i * gap + gap])
        print("===Data saved===")


if LOAD_ORIGINAL_DATA == False:
    # load embedding data
    print("Load reserved data.")
    with open(data_path + "train/" + "ori_train.npz", "rb") as f:
        ori_train_X = np.load(f)["x"]
        ori_train_Y = np.load(f)["y"]
    with open(data_path + "train/" + "cal_sv.npz", "rb") as f:
        cal_sv_X = np.load(f)["x"]
        cal_sv_Y = np.load(f)["y"]
    with open(data_path + "train/" + "raw_ori_train.npz", "rb") as f:
        raw_ori_train_X = np.load(f)["x"]
        raw_ori_train_Y = np.load(f)["y"]
    with open(data_path + "train/" + "raw_cal_sv.npz", "rb") as f:
        raw_cal_sv_X = np.load(f)["x"]
        raw_cal_sv_Y = np.load(f)["y"]

    for i in range(10):
        with open(data_path + 'train/pre_train_' + str(i) + '.npz', "rb") as f:
            x = np.load(f)["x"]
            y = np.load(f)["y"]
            if i == 0:
                pre_train_X = x
                pre_train_Y = y
            else:
                pre_train_X = np.concatenate((pre_train_X, x), axis=0)
                pre_train_Y = np.concatenate((pre_train_Y, y), axis=0)
        with open(data_path + 'raw_train/raw_train_' + str(i) + '.npz', "rb") as f:
            x = np.load(f)["x"]
            y = np.load(f)["y"]
            if i == 0:
                raw_pre_train_X = x
                raw_pre_train_Y = y
            else:
                raw_pre_train_X = np.concatenate((raw_pre_train_X, x), axis=0)
                raw_pre_train_Y = np.concatenate((raw_pre_train_Y, y), axis=0)


# calculate shapley values
k = 6

print("neighbour number:", k)
print("train shape: ", ori_train_X.shape, "calculate sv shape: ", cal_sv_X.shape)

if EMBED_SV == True:
    embed_knn_values, *_ = old_knn_shapley(k, ori_train_X, cal_sv_X, ori_train_Y, cal_sv_Y)
    np.savez_compressed(result_path + 'tiny_embed_knn.npz', knn=embed_knn_values)

    # old_fc1_knn_values, fc1_scores, fc1_false = loo_knn_shapley(k, train_X[:train_num], train_X[train_num:train_num+cal_num], train_Y[:train_num], train_Y[train_num:train_num+cal_num])
    # print("loo knn score on embedding data: ", fc1_scores)
    # np.savez_compressed(result_path + '_embed_loo_knn.npz', loo_knn=old_fc1_knn_values, score=fc1_scores, false=fc1_false)

if RAW_SV == True:
    knn_values, *_ = old_knn_shapley(k, raw_ori_train_X, raw_cal_sv_X, raw_ori_train_Y, raw_cal_sv_Y)
    np.savez_compressed(result_path + 'tiny_raw_knn.npz', knn=knn_values)

    # old_fc1_knn_values, fc1_scores, fc1_false = loo_knn_shapley(k, raw_train_X[:train_num], raw_train_X[train_num:train_num+cal_num], raw_train_Y[:train_num], raw_train_Y[train_num:train_num+cal_num])
    # print("loo knn score on embedding data: ", fc1_scores)
    # np.savez_compressed(result_path + 'raw_loo_knn.npz', loo_knn=old_fc1_knn_values, score=fc1_scores, false=fc1_false)


if DEEP_SV == True:
    if CAL_DEEP_FEATURES == True:
        raw_train_X = torch.from_numpy(raw_train_X).contiguous().view(-1, 3,64,64)
        raw_train_Y = torch.from_numpy(raw_train_Y).view(-1,).long()
        batch_size = 128
        epochs = 10
        model = model.to(device)
        plot_tiny_train("train", model, device, raw_train_X[:train_num], raw_train_Y[:train_num], optimizer, criterion, scheduler, batch_size, epochs)
        acc, _ = plot_tiny_train("val", model, device, raw_train_X[:train_num], raw_train_Y[:train_num], optimizer, criterion, scheduler, batch_size, epochs=1)
        print("Model for deep features acc: ", acc)

        deep_features = []
        for inputs, labels in batch(raw_train_X, raw_train_Y, batch_size):
            inputs = inputs.to(device)
            labels = labels.to(device)
            deep_feature, _ = model(inputs)
            deep_features.append(deep_feature.view(deep_feature.size(0), -1).cpu().detach().numpy())
        deep_features_X = np.concatenate(deep_features)
        deep_features_Y = raw_train_Y
        # print("deep features shape: ", deep_features.shape)
        # save deep features
        for i in range(10):
            np.savez_compressed(data_path + "deep_features/" + "df_" + str(i) + ".npz", x=deep_features_X[i * 10000:i * 10000 + 10000], y=raw_train_Y[i * 10000:i * 10000 + 10000])
    else:
        print("Load deep features data.")
        for i in range(10):
            with open(data_path + 'deep_features/df_' + str(i) + '.npz', "rb") as f:
                x = np.load(f)["x"]
                y = np.load(f)["y"]
                if i == 0:
                    deep_features_X = x
                    deep_features_Y = y
                else:
                    deep_features_X = np.concatenate((deep_features_X, x), axis=0)
                    deep_features_Y = np.concatenate((deep_features_Y, y), axis=0)

        deep_knn_values, *_ = old_knn_shapley(k, deep_features_X[:train_num], deep_features_X[train_num:train_num + cal_num], deep_features_Y[:train_num], deep_features_Y[train_num:train_num + cal_num])
        np.savez_compressed(result_path + 'tiny_deep_features_knn.npz', knn=deep_knn_values)

    old_fc1_knn_values, fc1_scores, fc1_false = loo_knn_shapley(k, deep_features_X[:train_num], deep_features_X[train_num:train_num+cal_num], deep_features_Y[:train_num], deep_features_Y[train_num:train_num+cal_num])
    print("deep features loo knn score on embedding data: ", fc1_scores)
    np.savez_compressed(result_path + 'tiny_deep_features_loo_knn.npz', loo_knn=old_fc1_knn_values, score=fc1_scores, false=fc1_false)
