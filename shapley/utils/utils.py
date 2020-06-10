import argparse
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from collections import OrderedDict
from scipy.misc import *
import torch
import cv2
import acoustics
import h5py
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils
import torch.optim as optim
import torch.nn as nn
import glob
import re
from PIL import Image


class CelebaDataset(Dataset):
    def __init__(self, label_file, root_dir, transform=None):
        # root_dir = "data/celeba/img_align_celeba/"
        self.labels, self.image_idxs = self.load_labels(label_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.image_idxs[idx])
        image = io.imread(img_name)
#         image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC)
        sample = {'image': image, 'label': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample
        
    def load_labels(self, label_file):
        # label_file="list_attr_celeba.csv"
        dir_anno = "data/celeba/"
        file = open(dir_anno + label_file, 'r')
        texts = file.read().split("\n")
        file.close()
        col_names = texts[0].split(",")
        Male_idx = col_names.index("Male")
        gender_list = []
        image_index_list = []
        for txt in texts[1:]:
            image_index_list.append(txt.split(',')[0])
            if txt.split(',')[Male_idx] == '1':
                gender_list.append(np.array(1))
            elif txt.split(',')[Male_idx] == '-1':
                gender_list.append(np.array(0))
        print(gender_list[:5], len(gender_list))
        gener_list = np.array(gender_list)
        return gender_list, image_index_list
                                
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        return {'image': img, 'label': labels}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, labels = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(labels)}

class MNIST():
    def __init__(self, one_hot=True, shuffle=False, by_label=False):
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(one_hot, by_label)
        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_test.shape[0]
        if shuffle: self.shuffle_data()

    def load_data(self, one_hot, by_label):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.reshape(x_train, [-1, 28, 28, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = np.reshape(x_test, [-1, 28, 28, 1])
        x_test = x_test.astype(np.float32) / 255

        if by_label:
            ind_train = np.argsort(y_train)
            ind_test = np.argsort(y_test)
            x_train, y_train = x_train[ind_train], y_train[ind_train]
            x_test, y_test = x_test[ind_test], y_test[ind_test]


        if one_hot:
            # convert to one-hot labels
            y_train = tf.keras.utils.to_categorical(y_train)
            y_test = tf.keras.utils.to_categorical(y_test)

        return x_train, y_train, x_test, y_test


    def shuffle_data(self):
        ind = np.random.permutation(self.num_train)
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]


class CIFAR10():
    def __init__(self, one_hot=True, shuffle=False):
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(one_hot)
        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_test.shape[0]

        if shuffle: self.shuffle_data()

    def load_data(self, one_hot):
        cifar = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar.load_data()
        # x_train.shape = (50000, 32, 32, 3), range = [0, 255]
        # y_train.shape = (50000, 1)

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

        if one_hot:
            # convert to one-hot labels
            y_train = tf.keras.utils.to_categorical(y_train)
            y_test = tf.keras.utils.to_categorical(y_test)

        return x_train, y_train, x_test, y_test


    def shuffle_data(self):
        ind = np.random.permutation(self.num_train)
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]



class Logger:
    def __init__(self, name='model', fmt=None, base="./logs"):
        self.handler = True
        self.scalar_metrics = OrderedDict()
        self.fmt = fmt if fmt else dict()
        if not os.path.exists(base): os.makedirs(base)
        self.path = os.path.join(base, name + "_" + str(time.time()))
        self.logs = self.path + '.csv'
        self.output = self.path + '.out'


        def prin(*args):
            str_to_write = ' '.join(map(str, args))
            with open(self.output, 'a') as f:
                f.write(str_to_write + '\n')
                f.flush()

            print(str_to_write)
            sys.stdout.flush()

        self.print = prin

    def add_scalar(self, t, key, value):
        if key not in self.scalar_metrics:
            self.scalar_metrics[key] = []
        self.scalar_metrics[key] += [(t, value)]

    def iter_info(self, order=None):
        names = list(self.scalar_metrics.keys())
        if order:
            names = order
        values = [self.scalar_metrics[name][-1][1] for name in names]
        t = int(np.max([self.scalar_metrics[name][-1][0] for name in names]))
        fmt = ['%s'] + [self.fmt[name] if name in self.fmt else '.1f' for name in names]

        if self.handler:
            self.handler = False
            self.print(tabulate([[t] + values], ['epoch'] + names, floatfmt=fmt))
        else:
            self.print(tabulate([[t] + values], ['epoch'] + names, tablefmt='plain', floatfmt=fmt).split('\n')[1])

    def save(self):
        result = None
        for key in self.scalar_metrics.keys():
            if result is None:
                result = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')
            else:
                df = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')
                result = result.join(df, how='outer')
        result.to_csv(self.logs)

        self.print('The log/output have been saved to: ' + self.path + ' + .csv/.out')

class ImageNet():
    def __init__(self, path, one_hot=True, shuffle=False):
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(path, one_hot)
        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_test.shape[0]
        if shuffle: self.shuffle_data()
        
        
    def load_data(self, path, one_hot):
        dog_fish = np.load(os.path.join(path, 'dataset_dog-fish_train-900_test-300.npz'))
        x_test = dog_fish[dog_fish.files[0]]
        x_train = dog_fish[dog_fish.files[1]]
        y_train = dog_fish[dog_fish.files[2]]
        y_test = dog_fish[dog_fish.files[3]]
        
        
        if one_hot:
            # convert to one-hot labels
            y_train = tf.keras.utils.to_categorical(y_train)
            y_test = tf.keras.utils.to_categorical(y_test)
        return x_train, y_train, x_test, y_test
   
    def shuffle_data(self):
        ind = np.random.permutation(self.num_train)
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]    
        
def add_noise(data, bs, target_snr, noise_type):
    if noise_type == 'white':
        noise = acoustics.generator.white(bs*28*28).reshape(28, 28, bs)
    if noise_type == 'pink':
        noise = acoustics.generator.pink(bs*28*28).reshape(28, 28, bs)
    if noise_type == 'Violet':
        noise = acoustics.generator.violet(bs*28*28).reshape(28, 28, bs)
        
   

    print ('data shape = ', data.shape)
    average = np.mean(data)
    std = np.std(noise)
    current_snr = average/std
    noise = noise * (current_snr/ target_snr)
    data = data + noise
    return data
        
def test_mnist():
    print ("Testing MNIST dataloader...")
    data = MNIST()
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    data = MNIST(one_hot=False)
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print (data.y_train[0:10])
    data = MNIST(shuffle=True, one_hot=False)
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print (data.y_train[0:10])
    data = MNIST(one_hot=False)
    fig=plt.figure(figsize=(8, 8))
    for i in range(1, 6):
#         img = data.x_train[i].reshape(1,28,28).transpose([1, 2, 0])
        
        img = data.x_train[i]
        img = add_noise(img, 1, 0.2, 'white')
        fig.add_subplot( 1, 5, i)
        plt.imshow(img.squeeze())
    plt.show()

def test_cifar10():
    print ("Testing CIFAR10 dataloader...")
    data = CIFAR10()
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    data = CIFAR10(one_hot=False)
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print (data.y_train[0:10])
    data = CIFAR10(shuffle=True, one_hot=False)
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print (data.y_train[0:10])
    fig=plt.figure(figsize=(8, 8))
    for i in range(1, 6):
        img = data.x_train[i] * 255
        fig.add_subplot( 1, 5, i)
        plt.imshow(img.astype(np.uint8))
    plt.show()

    
def test_imagenet():
    print("Testing ImageNet dataloader...")
    data = ImageNet('./data')
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    data = ImageNet(path='./data', one_hot=False)
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print (data.y_train[0:10])
    data = ImageNet(path='./data', shuffle=True, one_hot=False)
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print (data.y_train[0:10]) 
    fig=plt.figure(figsize=(8, 8))
    for i in range(1, 6):
#         img = data.x_train[i].reshape(3,299,299).transpose(1,2,0).astype("float")
        img = data.x_train[i] * -255
        fig.add_subplot( 1, 5, i)
        plt.imshow((img.squeeze()* 255).astype(np.uint8), interpolation='nearest')
    plt.show()
    
    
def train(model, device, x_train, y_train, batch_size, optimizer, criterion, n_epochs):
    model.train()
    for epoch in tqdm_notebook(range(n_epochs), desc = 'Epochs'):
#         print("epoch model.fc.weight:")
#         print(epoch, model.fc.weight)
        for X, y in batch(x_train, y_train, batch_size):  
            X, y = X.to(device).float(), y.to(device)
#             print(X.shape, y.shape)
            optimizer.zero_grad()
#             y_pred = model(X)
            *_, y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
#             for param in model.parameters():
#                 print(param.grad.data.sum())
            optimizer.step()
#         if(n_epochs > 4):
#             if(epoch % int(n_epochs/4) == 0):
#                 print(f'Train epoch {epoch}: Loss: {loss.item():7.4f}')

def evaluate(model, device, x_test, y_test, batch_size, criterion):
    model.eval()
    test_set_size = len(x_test)
    correct_answers = 0
    sum_loss = 0
    with torch.no_grad():
        for X, y in batch(x_test, y_test, batch_size):
            X, y = X.to(device).float(), y.to(device)
            *_, y_pred = model(X)
#             y_pred = model(X)
            
            class_pred = y_pred.argmax(dim = 1)
            correct_answers += (y == class_pred).float().sum().item()
            sum_loss += criterion(y_pred, y).item()
    accuracy = correct_answers / test_set_size
    average_loss = sum_loss / len(x_test)
    
    return accuracy, average_loss    

def evaluate_adv(model, device, x_test, y_test, batch_size, criterion):
    model.eval()
    test_set_size = len(x_test)
    correct_answers = 0
    sum_loss = 0
    idx = 0
    idxs = []
    falses = []
    ground_truths = []
    with torch.no_grad():
        for X, y in tqdm_notebook(batch(x_test, y_test, batch_size), total = int(len(x_test)/batch_size)):
            X, y = X.to(device).float(), y.to(device)
            *_, y_pred = model(X)
            class_pred = y_pred.argmax(dim = 1)
            correct_answers += (y == class_pred).float().sum().item()
#             print(y)
#             print(class_pred)
            if( y != class_pred):
                idxs.append(idx)
                falses.append(class_pred)
                ground_truths.append(y)
            idx += 1
            sum_loss += criterion(y_pred, y).item()
    accuracy = correct_answers / test_set_size
    average_loss = sum_loss / len(x_test)
    
    return accuracy, average_loss, falses, ground_truths, idxs

def knn_shapley(K, trainX, valX, trainy, valy):        
    N = trainX.shape[0]
    M = valX.shape[0]
    c = 1
#     value = np.zeros(N)
    value = [[] for i in range(N) ]
    scores = []
    false_result_idxs = []
    for i in tqdm_notebook(range(M), total=M, leave=False):
        X = valX[i]
        y = valy[i]

        s = np.zeros(N)
        diff = (trainX - X).reshape(N, -1) # calculate the distances between valX and every trainX data point
        dist = np.einsum('ij, ij->i', diff, diff) # output the sum distance
        idx = np.argsort(dist) # ascend the distance
        ans = trainy[idx]

        # calculate test performance
        score = 0.0
        
        for j in range(min(K, N)):
            score += float(ans[j] == y)
        if(score > min(K, N)/2):
            scores.append(1)
        else:
            scores.append(0)
            false_result_idxs.append(i)
        
        s[idx[N - 1]] = float(ans[N - 1] == y)*c / N
        cur = N - 2
        for j in range(N - 1):
            s[idx[cur]] = s[idx[cur + 1]] + float(int(ans[cur] == y) - int(ans[cur + 1] == y))*c / K * (min(cur, K - 1) + 1) / (cur + 1)
            cur -= 1
        
        for j in range(N):
            value[j].append(s[j])
#     for i in range(N):
#         value[i] /= M  
    return value, np.mean(scores), false_result_idxs

def old_knn_shapley(K, trainX, valX, trainy, valy):        
    N = trainX.shape[0]
    M = valX.shape[0]
    c = 1
    value = np.zeros(N)
#     value = [[] for i in range(N) ]
    scores = []
    false_result_idxs = []
    for i in tqdm_notebook(range(M), total=M, leave=False):
        X = valX[i]
        y = valy[i]

        s = np.zeros(N)
        diff = (trainX - X).reshape(N, -1) # calculate the distances between valX and every trainX data point
        dist = np.einsum('ij, ij->i', diff, diff) # output the sum distance
        idx = np.argsort(dist) # ascend the distance
        ans = trainy[idx]

        # calculate test performance
        score = 0.0
        
        for j in range(min(K, N)):
            score += float(ans[j] == y)
        if(score > min(K, N)/2):
            scores.append(1)
        else:
            scores.append(0)
            false_result_idxs.append(i)
        
        s[idx[N - 1]] = float(ans[N - 1] == y)*c / N
        cur = N - 2
        for j in range(N - 1):
            s[idx[cur]] = s[idx[cur + 1]] + float(int(ans[cur] == y) - int(ans[cur + 1] == y))*c / K * (min(cur, K - 1) + 1) / (cur + 1)
            cur -= 1
        
        for j in range(N):
            value[j] += s[j]
    for i in range(N):
        value[i] /= M  
    return value, np.mean(scores), false_result_idxs



def loo_knn_shapley(K, trainX, valX, trainy, valy):        
    N = trainX.shape[0]
    M = valX.shape[0]
    value = np.zeros(N)
    scores = []
    false_result_idxs = []
    for i in tqdm_notebook(range(M), total=M, leave=False):
        X = valX[i]
        y = valy[i]

        s = np.zeros(N)
        diff = (trainX - X).reshape(N, -1) # calculate the distances between valX and every trainX data point
        dist = np.einsum('ij, ij->i', diff, diff) # output the sum distance
        idx = np.argsort(dist) # ascend the distance
        ans = trainy[idx]
#         print(y, ans[:10])

        # calculate test performance
        score = 0.0
        
        for j in range(min(K, N)):
            score += float(ans[j] == y)
        if(score > min(K, N)/2):
            scores.append(1)
        else:
            scores.append(0)
            false_result_idxs.append(i)
            
        ### calculate LOO KNN values and do not concern the situation that K > N
        for j in range(N):
            if j in idx[:K]:
#                 print(int(ans[j] == y), int(ans[K] == y))
#                 print(y, j, ans[j], K, ans[K])
                s[j] = float(int(trainy[j] == y) - int(trainy[K] == y)) / K
            else:
                s[j] = 0
        
        
        for j in range(N):
            value[j] += s[j]
    for i in range(N):
        value[i] /= M  
    return value, np.mean(scores), false_result_idxs



def batch(x_batch, y_batch, batch_size=1):
    l = len(x_batch)
    for ndx in range(0, l, batch_size):
        yield x_batch[ndx:min(ndx + batch_size, l)], y_batch[ndx:min(ndx + batch_size, l)]

def print_img(img):
    plt.imshow(img.squeeze())
    plt.show()
    
def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

def h5load(path):
    # data means x, target means y
    if(os.path.exists(path)):
        with h5py.File(path, 'r') as hf:
            X_tr = hf.get('data')[:]
            y_tr = hf.get('target')[:]
            return X_tr, y_tr
        
def h5save(path, x, y):
    if(os.path.exists(path)):
        print("Already existed")
        return
    else:
        with h5py.File(path, 'w') as hf:
            hf.create_dataset("data",  data=x, compression="gzip", compression_opts=9)
            print("Data saved!")
            hf.create_dataset("target", data=y, compression="gzip", compression_opts=9)
            print("Target saved!")
    
        return

def cw_l2_attack(model, images, labels, device, targeted=False, c=1e-4, kappa=1, max_iter=1000, learning_rate=0.01) :
    images = images.to(device)     
    labels = labels.to(device)
    # Define f-function
    def f(x) :
        *_, outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
#         print(i,j)
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)   
    w = torch.zeros_like(images, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate)
    prev = 1e10
    for step in range(max_iter) :
        a = 1/2*(nn.Tanh()(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))
        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward()
#         print(cost)
        optimizer.step()
        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost      
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')
    attack_images = 1/2*(nn.Tanh()(w) + 1)
    return attack_images

def load_filenames_labels(mode):
    """Gets filenames and labels
    Args:
      mode: 'train' or 'val'
      (Directory structure and file naming different for
      train and val datasets)
    Returns:
      list of tuples: (jpeg filename with path, label)
    """
    label_dict, class_description = build_label_dicts()
    filenames_labels = []
    if mode == 'train':
        filenames = glob.glob('data/tiny-imagenet-200/train/*/images/*.JPEG')
        for filename in filenames:
            match = re.search(r'n\d+', filename)
            label = str(label_dict[match.group()])
            filenames_labels.append((filename, label))
    elif mode == 'val':
        with open('data/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                filename = 'data/tiny-imagenet-200/val/images/' + split_line[0]
                label = str(label_dict[split_line[1]])
                filenames_labels.append((filename, label))

    return filenames_labels

def build_label_dicts():
    """Build look-up dictionaries for class label, and class description
  Class labels are 0 to 199 in the same order as 
    tiny-imagenet-200/wnids.txt. Class text descriptions are from 
    tiny-imagenet-200/words.txt
  Returns:
    tuple of dicts
      label_dict: 
        keys = synset (e.g. "n01944390")
        values = class integer {0 .. 199}
      class_desc:
        keys = class integer {0 .. 199}
        values = text description from words.txt
    """
    label_dict, class_description = {}, {}
    with open('data/tiny-imagenet-200/wnids.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            synset = line[:-1]  # remove \n
            label_dict[synset] = i
        with open('data/tiny-imagenet-200/words.txt', 'r') as f:
            for i, line in enumerate(f.readlines()):
                synset, desc = line.split('\t')
                desc = desc[:-1]  # remove \n
                if synset in label_dict:
                    class_description[label_dict[synset]] = desc

    return label_dict, class_description

def load_tinyImagenet(dataset):
    dim = np.zeros((64,64))
    imgs = []
    labels = []
    for path, label in dataset:
        img=np.array(Image.open(path)) /255.0
#         print(path, len(img.shape))
        if(len(img.shape) != 3):
            img = np.stack((img, dim, dim), axis=2)       
        imgs.append(img)
        labels.append(int(label))
    return imgs, labels