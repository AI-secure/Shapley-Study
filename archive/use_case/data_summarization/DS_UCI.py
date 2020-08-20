import math, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

from shapley.models.uci import *
from shapley.utils.shap_utils import *
from shapley.utils.utils import *

# remove new data to the train set and draw the picture
def plot_summarization(loo_pre_idx, tmc_pre_idx, g_values, knn_values, loo_knn_values, kmin, kmax, kinterval, x_train, y_train, x_test, y_test, HtoL=False):
    sns.set()
    plt.figure()
    # set the model deterministic
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    plt.xlabel('Fraction of train data removed (%)')
    plt.ylabel('Prediction accuracy (%)')

    tmc_accs = []
    knn_accs = [[] for _ in range(len(knn_values))]
    loo_knn_accs = [[] for _ in range(len(knn_values))]
    marks = ['o-', '*-', 's-', 'v-', '^-', '.-', '1-', '2-']
    colors = ['b', 'g', 'r', 'y', 'c', 'm', 'olive', 'pink']
    labels = [i for i in range(kmin, kmax, kinterval)]
    model = 'uci'
    count = int(len(x_train)/2)
    interval = int(count*0.02)
    x = np.arange(0, count, interval)/len(x_train)
    if(HtoL == True):
        print("removing data from Highest to Lowest!")
        knn_pre_idx = np.flip(knn_pre_idx, 1)
        tmc_pre_idx = np.flip(tmc_pre_idx, 0)
        loo_pre_idx = np.flip(loo_pre_idx, 0)
    else:
        print("removing data from Lowest to Highest!")
    # G Shapley
    print("G")
    g_accs = []
    idxs = np.argsort(g_values)
    keep_idxs = idxs.tolist()
    for j in range(0, count, interval):
        if len(keep_idxs) == len(x_train):
            x_train_keep, y_train_keep = x_train, y_train
        else:
            x_train_keep, y_train_keep = x_train[keep_idxs], y_train[keep_idxs]
        clf_g = return_model(model)
        clf_g.fit(x_train_keep, y_train_keep)
        acc = clf_g.score(x_test, y_test)
        print(len(keep_idxs), acc)
        g_accs.append(acc)
        keep_idxs = keep_idxs[interval:]
    print("G:", g_accs)
    plt.plot(x, g_accs, '-', label='G Shapley', color='olive')


    # TMC Shapley
    print("TMC")
    tmc_accs = []
    idxs = np.argsort(tmc_pre_idx)
    keep_idxs = idxs.tolist()
    for j in range(0, count, interval):
        if len(keep_idxs) == len(x_train):
            x_train_keep, y_train_keep = x_train, y_train
        else:
            x_train_keep, y_train_keep = x_train[keep_idxs], y_train[keep_idxs]
        clf_tmc = return_model(model)
        clf_tmc.fit(x_train_keep, y_train_keep)
        acc = clf_tmc.score(x_test, y_test)
        print(len(keep_idxs), acc)
        tmc_accs.append(acc)
        keep_idxs = keep_idxs[interval:]
    print("TMC:", tmc_accs)
    plt.plot(x, tmc_accs, '-', label='TMC Shapley', color='olive')

    # Loo Shapley
    print("Loo")
    loo_accs = []
    idxs = np.argsort(loo_pre_idx)
    keep_idxs = idxs.tolist()
    for j in range(0, count, interval):
        if len(keep_idxs) == len(x_train):
            x_train_keep, y_train_keep = x_train, y_train
        else:
            x_train_keep, y_train_keep = x_train[keep_idxs], y_train[keep_idxs]
        clf_loo = return_model(model)
        clf_loo.fit(x_train_keep, y_train_keep)
        acc = clf_loo.score(x_test, y_test)
        print(len(keep_idxs), acc)
        loo_accs.append(acc)
        keep_idxs = keep_idxs[interval:]
    print("LOO: ", loo_accs)
    plt.plot(x, loo_accs, '^-', label='LOO', color='pink')

    # Knn Shapley
    print("KNN")
    for i in range(len(knn_values)):
        idxs = np.argsort(knn_values[i])
        keep_idxs = idxs.tolist()
        x = np.arange(0, count, interval)/len(x_train)

        for j in tqdm_notebook(range(0, count, interval), total=int(count/interval), leave=False):
            if len(keep_idxs) == len(x_train):
                x_train_keep, y_train_keep = x_train, y_train
            else:
                x_train_keep, y_train_keep = x_train[keep_idxs], y_train[keep_idxs]
            clf_knn = return_model(model)
            clf_knn.fit(x_train_keep, y_train_keep)
            acc = clf_knn.score(x_test, y_test)
            acc_t = clf_knn.score(x_train_keep, y_train_keep)

            knn_accs[i].append(acc)
            keep_idxs = keep_idxs[interval:]
            clf_knn = 0
    for i in range(0, len(knn_values)):
        print(x, knn_accs[i])
        plt.plot(x, knn_accs[i], '-', label="Knn Shapley, K="+str(labels[i]), color=colors[i])

    # LOO Knn Shapley
    print("LOO KNN")
    for i in range(len(loo_knn_values)):
        idxs = np.argsort(loo_knn_values[i])
        keep_idxs = idxs.tolist()
        x = np.arange(0, count, interval)/len(x_train)

        for j in tqdm_notebook(range(0, count, interval), total=int(count/interval), leave=False):
            if len(keep_idxs) == len(x_train):
                x_train_keep, y_train_keep = x_train, y_train
            else:
                x_train_keep, y_train_keep = x_train[keep_idxs], y_train[keep_idxs]
            clf_knn = return_model(model)
            clf_knn.fit(x_train_keep, y_train_keep)
            acc = clf_knn.score(x_test, y_test)
            loo_knn_accs[i].append(acc)
            keep_idxs = keep_idxs[interval:]

    for i in range(0, len(knn_values)):
        print(x, loo_knn_accs[i])
        plt.plot(x, loo_knn_accs[i], '-', label="LOO Knn Shapley, K="+str(labels[i]), color=colors[i+2])

    # random solution
    times = 5
    all_rand_accs = []
    print("random")
    for time in range(times):
        print(time)
        random_accs = []
        keep_idxs = np.arange(0, len(x_train))
        random.shuffle(keep_idxs)
        for j in range(0, count, interval):
            if len(keep_idxs) == len(x_train):
                x_train_keep, y_train_keep = x_train, y_train
            else:
                x_train_keep, y_train_keep = x_train[keep_idxs], y_train[keep_idxs]
            clf_random = return_model(model)
            clf_random.fit(x_train_keep, y_train_keep)
            acc = clf_random.score(x_test, y_test)
            random_accs.append(acc)
            keep_idxs = keep_idxs[interval:]
        all_rand_accs.append(random_accs)
    all_rand_accs = np.mean(all_rand_accs, 0)
    print("random:", all_rand_accs.tolist())
    plt.plot(x, all_rand_accs, '-', label='random', color='red')
    plt.legend()
    plt.tight_layout()
    if(HtoL == True):
        plt.savefig('knnH-L.png')
    else:
        plt.savefig('knnL-H.png')
    return



column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation',
                'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'income']

train = pd.read_csv('./data/uci/data.train', sep=",\s", header=None, names = column_names, engine = 'python')
test = pd.read_csv('./data/uci/data.test', sep=",\s", header=None, names = column_names, engine = 'python')
test['income'].replace(regex=True,inplace=True,to_replace=r'\.',value=r'')

adult = pd.concat([test,train])
adult.reset_index(inplace = True, drop = True)
for col in set(adult.columns) - set(adult.describe().columns):
    adult[col] = adult[col].astype('category')

# removing ? and Nan
adult = adult.dropna(how='any')
adult = adult[(adult.astype(str) != '?').all(axis=1)]

# Data Prep
adult_data = adult.drop(columns = ['income'])
adult_label = adult.income

le = preprocessing.LabelEncoder()
le.fit(adult_label)
adult_label = le.transform(adult_label)

adult_cat_1hot = pd.get_dummies(adult_data.select_dtypes('category'))
adult_non_cat = adult_data.select_dtypes(exclude = 'category')
adult_data_1hot = pd.concat([adult_non_cat, adult_cat_1hot], axis=1, join='inner')

idx0 = np.where(adult_label == 0)[0]
idx1 = np.where(adult_label == 1)[0]
print(len(idx0),len(idx1))

adult_data_1hot = adult_data_1hot.to_numpy()

train_size = 500
test_size = 250
heldout_size = 500

# print(adult_data_1hot(idx0[:train_size]))
train_data = np.concatenate((adult_data_1hot[idx0[:train_size]], adult_data_1hot[idx1[:train_size]]), 0)
train_label = np.concatenate((adult_label[idx0[:train_size]], adult_label[idx1[:train_size]]), axis=0)
test_data = np.concatenate((adult_data_1hot[idx0[train_size:train_size+test_size]], adult_data_1hot[idx1[train_size:train_size+test_size]]),0)
test_label = np.concatenate((adult_label[idx0[train_size:train_size+test_size]], adult_label[idx1[train_size:train_size+test_size]]), 0)
heldout_data = np.concatenate((adult_data_1hot[idx0[train_size+test_size:train_size+test_size+heldout_size]], adult_data_1hot[idx1[train_size+test_size:train_size+test_size+heldout_size]]), 0)
heldout_label = np.concatenate((adult_label[idx0[train_size+test_size:train_size+test_size+heldout_size]], adult_label[idx1[train_size+test_size:train_size+test_size+heldout_size]]), 0)

print(train_data.shape, heldout_data.shape, test_data.shape)

# Normalization
scaler = StandardScaler()

# Fitting only on training data
scaler.fit(train_data)
train_data = scaler.transform(train_data)

# Applying same transformation to test data
test_data = scaler.transform(test_data)
heldout_data = scaler.transform(heldout_data)

model = "uci"
clf = return_model(model)
# xx = 1000
clf.fit(train_data, train_label)
# pre_t = clf.predict(test_data)
directory = './temp_ds_uci'

test_acc = clf.score(heldout_data, heldout_label)
num_test = test_data.shape[0]



#data preparation
batch_size = 1024
epochs = 400

x_train = torch.from_numpy(train_data).contiguous().view(-1, 254)
y_train = torch.from_numpy(train_label).view(-1,).long()
print("train_size:", x_train.shape)
x_test = torch.from_numpy(test_data).contiguous().view(-1, 254)
y_test = torch.from_numpy(test_label).view(-1,).long()
print("test_size:", x_test.shape)
x_heldout = torch.from_numpy(heldout_data).contiguous().view(-1, 254)
y_heldout = torch.from_numpy(heldout_label).view(-1,).long()
print("heldout_size:", x_heldout.shape)

device = torch.device('cuda')
uci = UCI().to(device)
optimizer = optim.Adam(uci.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train(uci, device, x_train, y_train, batch_size, optimizer, criterion, epochs)
accuracy, avg_loss = evaluate(uci, device, x_train, y_train, batch_size, criterion)
print(f'[Train] Accuracy: {100 * accuracy:5.2f}%, loss: {avg_loss:7.4f}')
accuracy, avg_loss = evaluate(uci, device, x_heldout, y_heldout, batch_size, criterion)
print(f'[Test] Accuracy: {100 * accuracy:5.2f}%, loss: {avg_loss:7.4f}')

deep_f_train = []
deep_f_test = []
targets = []
for X, y in batch(x_train, y_train, batch_size):
    X = X.to(device).float()
    fc3, y_pre = uci(X)
    deep_f_train.append(fc3.view(fc3.size(0), -1).cpu().detach().numpy())

for X, y in batch(x_test, y_test, batch_size):
    X = X.to(device).float()
    fc3, y_pre = uci(X)
    deep_f_test.append(fc3.view(fc3.size(0), -1).cpu().detach().numpy())

deep_f_train = np.concatenate(deep_f_train) # deep features are not normalized
deep_f_test = np.concatenate(deep_f_test) # deep features are not normalized
print(deep_f_train.shape, deep_f_test.shape)



kmin = 5
kmax = 6
kinterval = 5
fc1_knn_values = [[] for _ in range(math.ceil((kmax-kmin)/kinterval))] # deep features
loo_fc1_knn_values = [[] for _ in range(math.ceil((kmax-kmin)/kinterval))] # deep features

# t = 10000
for i, k in enumerate(range(kmin, kmax, kinterval)):
    print("neighbour number:", k)
    loo_fc1_knn_values[i],*_ = loo_knn_shapley(k, deep_f_train, deep_f_test, y_train, y_test)
    fc1_knn_values[i],*_ = old_knn_shapley(k, deep_f_train, deep_f_test, y_train, y_test)

plot_summarization(loo_values, tmc_values, g_values, fc1_knn_values, loo_fc1_knn_values, kmin, kmax, kinterval,
          x_train, y_train, x_heldout, y_heldout, False)

