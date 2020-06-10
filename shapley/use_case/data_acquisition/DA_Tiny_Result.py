import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from shapley.use_case.plot.plot_resnet import eval_resnet_acq_tiny_single


LOAD_RESERVED_DATA = True
LOAD_ORIGINAL_VAL_DATA = False
LOAD_SV = True
SAVE = False
data_path = "embedding_data/da/"
result_path = 'embedding_result/da/'
data_dir = "data/tiny-imagenet-200"


print("load val data: ", LOAD_ORIGINAL_VAL_DATA)
if LOAD_ORIGINAL_VAL_DATA == True:
    # load validation dataset
    data_transform = transforms.Compose([transforms.ToTensor()])
    image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transform)
    batch_size = len(image_datasets)
    val_dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=64)
    val_X, val_Y = next(iter(val_dataloader))
    # change dataloaders to numpy
    val_X = val_X.numpy() # (10000, 3, 64, 64)
    val_Y = val_Y.numpy()  # (10000, )
    # save validation data
    np.savez_compressed(data_path + "val/" + "validation.npz", x=val_X, y=val_Y)
else:
    with open(data_path + "val/" + "validation.npz", "rb") as f:
            val_X = np.load(f)["x"]
            val_Y = np.load(f)["y"]
print("val train shape: ", val_X.shape)

print("load reserved data: ", LOAD_RESERVED_DATA)
if LOAD_RESERVED_DATA == True:
    with open(data_path + "train/" + "ori_train.npz", "rb") as f:
        embed_train_X = np.load(f)["x"]
        embed_train_Y = np.load(f)["y"]
    # with open(data_path + "train/" + "cal_sv.npz", "rb") as f:
    #     cal_sv_X = np.load(f)["x"]
    #     cal_sv_Y = np.load(f)["y"]
    with open(data_path + "raw_train/" + "raw_ori_train.npz", "rb") as f:
        raw_train_X = np.load(f)["x"]
        raw_train_Y = np.load(f)["y"]
    # with open(data_path + "raw_train/" + "raw_cal_sv.npz", "rb") as f:
    #     raw_cal_sv_X = np.load(f)["x"]
    #     raw_cal_sv_Y = np.load(f)["y"]
    for i in range(10):
        with open(data_path + 'train/pre_train_' + str(i) + '.npz', "rb") as f:
            x = np.load(f)["x"]
            y = np.load(f)["y"]
            if i == 0:
                embed_pre_train_X = x
                embed_pre_train_Y = y
            else:
                embed_pre_train_X = np.concatenate((embed_pre_train_X, x), axis=0)
                embed_pre_train_Y = np.concatenate((embed_pre_train_Y, y), axis=0)
        with open(data_path + 'raw_train/raw_pre_train_' + str(i) + '.npz', "rb") as f:
            x = np.load(f)["x"]
            y = np.load(f)["y"]
            if i == 0:
                raw_pre_train_X = x
                raw_pre_train_Y = y
            else:
                raw_pre_train_X = np.concatenate((raw_pre_train_X, x), axis=0)
                raw_pre_train_Y = np.concatenate((raw_pre_train_Y, y), axis=0)
    print("train shape: ", raw_train_X.shape, "raw_pre_train shape: ", raw_pre_train_X.shape)

# load knn shapley values
if LOAD_SV == True:
    with open(result_path + 'tiny_embed_knn.npz', 'rb') as f:
        embed_knn_sv = np.load(f)["knn"]
    # with open(result_path + 'embed_loo_knn.npz', 'rb') as f:
    #     embed_loo_knn_sv = np.load(f)["loo_knn"]
    # with open(result_path + 'raw_knn.npz', 'rb') as f:
    #     raw_knn_sv = np.load(f)["knn"]
    # with open(result_path + 'raw_loo_knn.npz', 'rb') as f:
    #     raw_loo_knn_sv = np.load(f)["loo_knn"]
    # with open(result_path + 'deep_features_knn.npz', 'rb') as f:
    #     df_knn_sv = np.load(f)["knn"]
    # with open(result_path + 'deep_features_loo_knn.npz', 'rb') as f:
    #     df_loo_knn_sv = np.load(f)["loo_knn"]

print("embed knn sv shape: ", embed_knn_sv.shape)
# print("raw knn sv shape: ", raw_knn_sv.shape)
# print("knn sv shape: ", knn_sv.shape, "loo score: ", score, "loo knn sv shape: ", loo_knn_sv.shape)

# train random forest for embeddings
filted_embed_knn_sv_idxs = np.where(embed_knn_sv >= 0.0)[0]
embed_knn_sv = embed_knn_sv / np.linalg.norm(embed_knn_sv)

random_forest =  RandomForestRegressor(max_depth=100, n_estimators=50, random_state=666)
random_forest.fit(embed_train_X[filted_embed_knn_sv_idxs][:,:,0,0], embed_knn_sv[filted_embed_knn_sv_idxs])

embed_knn_pre_scores = random_forest.predict(embed_pre_train_X[:,:,0,0])


batch_size = 128
k = 6

sx_train = torch.from_numpy(raw_train_X).contiguous().view(-1, 3,64,64)
sy_train = torch.from_numpy(raw_train_Y).view(-1,).long()
print("train_size:", sx_train.shape)
sx_test = torch.from_numpy(val_X).contiguous().view(-1, 3,64,64)
sy_test = torch.from_numpy(val_Y).view(-1,).long()
print("test_size:", sx_test.shape)
sx_pre = torch.from_numpy(raw_pre_train_X).contiguous().view(-1, 3,64,64)
sy_pre = torch.from_numpy(raw_pre_train_Y).view(-1,).long()
print("pre_size:", sx_pre.shape)


HtoL = True
device_id = 2
x_ratio = 0.1

count = int(len(sx_pre))
interval = int(count * x_ratio)
x_arrange = np.arange(0, count, interval)

# eval_resnet_tiny(knn_sv, raw_knn_sv, k, sx_train, sy_train, sx_val, sy_val, batch_size, epochs=10, HtoL=HtoL, device_id=device_id)

# random_acc = eval_resnet_tiny_random(embed_knn_sv, sx_train, sy_train, sx_val, sy_val, batch_size, x_ratio, epochs=10, device_id=device_id)
# print("random acc: ", random_acc)

embed_acc = eval_resnet_acq_tiny_single("embed", embed_knn_pre_scores, sx_train, sy_train, sx_test, sy_test, sx_pre, sy_pre, batch_size, x_ratio, epochs=15, HtoL=HtoL, device_id=device_id)
# embed_loo_acc = eval_resnet_tiny_single("embed loo", embed_loo_knn_sv, sx_train, sy_train, sx_val, sy_val, batch_size, x_ratio, epochs=10, HtoL=HtoL, device_id=device_id)

# raw_acc = eval_resnet_tiny_single("raw", raw_knn_sv, sx_train, sy_train, sx_val, sy_val, batch_size, x_ratio, epochs=10, HtoL=HtoL, device_id=device_id)
# raw_loo_acc = eval_resnet_tiny_single("raw loo", raw_loo_knn_sv, sx_train, sy_train, sx_val, sy_val, batch_size, x_ratio, epochs=10, HtoL=HtoL, device_id=device_id)

# df_acc = eval_resnet_tiny_single("df", df_knn_sv, sx_train, sy_train, sx_val, sy_val, batch_size, x_ratio, epochs=10, HtoL=HtoL, device_id=device_id)
# df_loo_acc = eval_resnet_tiny_single("df loo", df_loo_knn_sv, sx_train, sy_train, sx_val, sy_val, batch_size, x_ratio, epochs=10, HtoL=HtoL, device_id=device_id)


print("embed_acc: ", embed_acc)
# print("embed_loo_acc: ", embed_loo_acc)
# print("raw_acc: ", raw_acc)
# print("raw_loo_knn_sv: ", raw_loo_acc)
# print("df_knn_sv: ", df_acc)
# print("df_loo_knn_sv: ", df_loo_acc)


print("x Arrange: ", x_arrange)
# eval_resnet_tiny(knn_sv, loo_knn_sv, k, sx_train, sy_train, sx_val, sy_val, batch_size, epochs=10, HtoL=True)
if SAVE == True:
    if HtoL == True:
        np.savez(result_path+'val_result_HtoL.npz', x=x_arrange, random=random_acc, embed_acc=embed_acc, embed_loo_knn_sv=embed_loo_knn_sv, raw_acc=raw_acc, raw_loo_knn_sv=raw_loo_acc, df_knn_sv=df_acc, df_loo_knn_sv=df_loo_acc)
        print("===val result saved===")
    elif HtoL == False:
        np.savez(result_path+'val_result_LtoH.npz', x=x_arrange, random=random_acc, embed_acc=embed_acc, embed_loo_knn_sv=embed_loo_knn_sv, raw_acc=raw_acc, raw_loo_knn_sv=raw_loo_acc, df_knn_sv=df_acc, df_loo_knn_sv=df_loo_acc)
        print("===val result saved===")
    else:
        print("error! no saved")

