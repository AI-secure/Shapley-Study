# pubfig 
from PIL import Image
import os
import os.path
import numpy as np
import sys

import torch.utils.data as data
from torchvision.datasets import ImageFolder


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class PUBFIG83(data.Dataset):
    def __init__(self, root='./pubfig_data/pubfig83-aligned', train=True, num_perclass=108, transform=None):
        self.root = root
        self.imgs_all = ImageFolder(root).imgs
        self.classes = ImageFolder(root).classes
        
        self.n_class = 83
        self.num_perclass = num_perclass if train else 10
        self.transform = transform
        self.loader = default_loader
        
        if not train:
            self.imgs_all.reverse()

        cnt = [0 for i in range(self.n_class)]
        self.imgs = []
        for img, lbl in self.imgs_all:
            if cnt[lbl] < self.num_perclass:
                cnt[lbl] += 1
                self.imgs.append((img, lbl))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.imgs)

def create_custom_mnist_ds(path, input_size=224):
    """Returns a tensor dataset for custom mnistData.npz"""
    numpy_data = np.load(path)
    mnist_x = numpy_data['x']
    mnist_y = numpy_data['y']
    x = torch.Tensor(mnist_x)
    y = torch.Tensor(mnist_y)
    y = y.type(torch.LongTensor)

    # reshape into BCHW format
    x = x.reshape(-1, 1, 28, 28)
    # expand to three channels for pretrained imageNet
    x = x.expand(-1, 3, -1, -1)
    # resize image to input_size
    x = F.interpolate(x, input_size)
    custom_mnist = data.TensorDataset(x, y)
    return custom_mnist

def create_usps_ds(path, train=True, input_size=224):
    """Creates a tensor dataset out of the USPS.h5 dataset"""
    
    choose = 'train'
    if not train:
        choose = 'test'
        
    with h5py.File(path, 'r') as hf:
        usps_data = hf.get(choose)
        usps_X = np.array(usps_data.get('data')[:])
        usps_y = usps_data.get('target')[:]
        
    x = torch.Tensor(usps_X)
    y = torch.Tensor(usps_y)
    y = y.type(torch.LongTensor)

    # reshape into BCHW format
    x = x.reshape(-1, 1, 16, 16)
    # expand to three channels for pretrained imageNet
    x = x.expand(-1, 3, -1, -1)
    # resize image to input_size
    x = F.interpolate(x, input_size)
    custom_usps = data.TensorDataset(x, y)
    return custom_usps