import argparse
from efficientnet_pytorch import EfficientNet
import os
import os.path as osp

import torch.nn as nn
from torchvision import datasets, transforms as T
import torchvision.models as models

from shapley.embedding.custom_loader import PUBFIG83, create_custom_mnist_ds, create_usps_ds
from shapley.embedding.extractor import MobileNetFeatureExtractor, VGGFeatureExtractor, EfficientNetExtractor, InceptionFeatureExtractor
from shapley.embedding.helper import create_embeddings


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor', type=str, 
        choices=['vgg11', 'mobilenet', 'resnet18', 'efficientnet', 'inception'])
    parser.add_argument('--dataset', type=str,
        choices=['mnist', 'fashionmnist', 'svhn', 'cifar', 'cifar_watermarked', 
                'pubfig', 'pubfig_watermarked', 'tinyimagenet', 'test_custom_mnist', 
                'train_custom_mnist', 'prediction_custom_mnist', 'train_usps_ds', 
                'test_usps_ds'])
    return parser.parse_args()


def get_extractor(extractor_name):
    if extractor_name == 'vgg11':
        vgg11 = models.vgg11(pretrained = True)
        return VGGFeatureExtractor(vgg11), 500

    if extractor_name == 'mobilenet':
        mobilenet = models.mobilenet_v2(pretrained = True)
        return MobileNetFeatureExtractor(mobilenet), 1000

    if extractor_name == 'resnet18':
        resnet18 = models.resnet18(pretrained = True)
        return nn.Sequential(*list(resnet18.children())[:-1]), 1000

    if extractor_name == 'efficientnet':
        efficientB7 = EfficientNet.from_pretrained('efficientnet-b7')
        return EfficientNetExtractor(efficientB7), 50

    if extractor_name == 'inception':
        inception = models.inception_v3(pretrained = True)
        return InceptionFeatureExtractor(inception), 1000

    raise NotImplementedError(f'extractor = {extractor_name} not implemented!')


def get_dataset(dataset_name):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    grey_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.Grayscale(3), T.ToTensor(), normalize])

    data_dir = 'data'
    if not osp.exists(data_dir): os.makedirs(data_dir)

    if dataset_name == 'mnist':
        return datasets.MNIST("./data/mnist", train=True, transform=grey_transform, target_transform=None, download=True)

    if dataset_name == 'fashionmnist':
        return datasets.FashionMNIST("./data/fashionmnist", train=True, transform=grey_transform, target_transform=None, download=True)

    if dataset_name == 'svhn':
        return datasets.SVHN("./data/svhn", split='train', transform=transform, target_transform=None, download=True)

    if dataset_name == 'cifar':
        return datasets.CIFAR10("./data/cifar", train=True, transform=transform, target_transform=None, download=True)

    if dataset_name == 'cifar_watermarked':
        return datasets.ImageFolder("./data/CIFAR_data/watermarked_ds", transform=transform)

    if dataset_name == 'pubfig':
        return PUBFIG83(root='./data/pubfig_data/pubfig83-aligned', train=True, num_perclass=108, transform=transform)

    if dataset_name == 'pubfig_watermarked':
        return PUBFIG83(root='./data/pubfig_data/watermarked_ds', train=True, num_perclass=108, transform=transform)

    if dataset_name == 'tinyimagenet':
        return datasets.ImageFolder("./data/tiny-imagenet-200/train", transform=transform)

    if dataset_name == 'test_custom_mnist':
        return create_custom_mnist_ds("./data/mnistForData/test.npz")

    if dataset_name == 'train_custom_mnist':
        return create_custom_mnist_ds("./data/mnistForData/train.npz")

    if dataset_name == 'prediction_custom_mnist':
        return create_custom_mnist_ds("./data/mnistForData/prediction.npz")

    if dataset_name == 'train_usps_ds':
        return create_usps_ds('./data/usps/usps.h5')

    if dataset_name == 'test_usps_ds':
        return create_usps_ds('./data/usps/usps.h5', train=False)

    raise NotImplementedError(f'dataset = {dataset_name} not implemented')


args = get_arguments()

extractor, storage_size = get_extractor(args.extractor)
dataset = get_dataset(args.dataset)
create_embeddings(extractor, dataset, args.extractor, 
                f"./all-features/{args.dataset}_{args.extractor}", storage_size=storage_size)