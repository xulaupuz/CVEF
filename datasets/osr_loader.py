# code in this file is adpated from
# https://github.com/iCGY96/ARPL
# https://github.com/wjun0830/Difficulty-Aware-Simulator

import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from .tools import *
# DATA_PATH = '/HOME/scw6ceg/run/ml/datasets'
DATA_PATH = 'C://mlcodes/datasets'
# DATA_PATH = '_'
SVHN_PATH = DATA_PATH + '/svhn/'
TINYIMAGENET_PATH = DATA_PATH + '/tiny_imagenet/'
CROOD_PATH = DATA_PATH + '/odin_ood/'
# CIFAR10_PATH = DATA_PATH
# CIFAR100_PATH = DATA_PATH

class CIFAR10_Filter(CIFAR10):
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(
            np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR10_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = predata(img_size)
        transform = test_transform(img_size)
        
        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory
        )

        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)        
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class CIFAR100_Filter(CIFAR100):
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(
            np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR100_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))
        print('Selected Labels: ', known)

        train_transform = predata(img_size)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        

class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """

    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)


class SVHN_OSR(object):
    def __init__(self, known, dataroot=SVHN_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = predata(img_size)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = SVHN_Filter(root=dataroot, split='train',
                               download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class Tiny_ImageNet_Filter(ImageFolder):
    def __init__(self,root,transform, known=None, is_val=False):
        super().__init__(root,transform=transform)
        self.is_val = is_val
        if is_val:
            self.annotations = self.read_val_annotations()
            self.classes = sorted(set(self.annotations.values()))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.imgs = self.make_val_dataset()
        if known is not None:
            self.__Filter__(known)

    def read_val_annotations(self):  # Used for tinyimagenet-val
        annotations_path = os.path.join(self.root, 'val_annotations.txt')
        annotations = {}
        with open(annotations_path, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                filename = parts[0]
                label = parts[1]
                annotations[filename] = label
        return annotations

    def make_val_dataset(self):
        images = []
        for filename, label in self.annotations.items():
            path = os.path.join(self.root, 'images', filename)
            images.append((path, self.class_to_idx[label]))
        return images

    def __Filter__(self, known):
        datas = self.imgs
        targets = [_[1] for _ in datas]
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot=TINYIMAGENET_PATH, use_gpu=True, batch_size=128, img_size=64, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print('Selected Labels: ', known)

        train_transform = predata(img_size)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(root=os.path.join(dataroot, 'tiny-imagenet-200', 'train'),transform=train_transform,known=self.known)
        # trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = Tiny_ImageNet_Filter(root=os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform=transform,known=self.known, is_val=True)
        # testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = Tiny_ImageNet_Filter(root=os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform=transform,known=self.unknown, is_val=True)
        # outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class Tiny_ImageNet_Crop(object):
    def __init__(self, dataroot=CROOD_PATH + 'ImagenetCrop', use_gpu=True, batch_size=128, img_size=32, options=None):
        transform =  test_transform(img_size)

        pin_memory = True if use_gpu else False
        testset = ImageFolder(dataroot, transform=transform)
        # testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        print('TinyImageNetCrop Test Num: ', len(testset))

class Tiny_ImageNet_Resize(object):
    def __init__(self, dataroot=CROOD_PATH + 'Imagenet_resize', use_gpu=True, batch_size=128, img_size=32, options=None):
        transform =  test_transform(img_size)
        pin_memory = True if use_gpu else False
        testset = ImageFolder(dataroot, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        print('TinyImageNetResize Test Num: ', len(testset))

class LSUN_Crop(object):
    def __init__(self, dataroot=CROOD_PATH + 'LSUN', use_gpu=True, batch_size=128, img_size=32, options=None):
        transform =  test_transform(img_size)

        pin_memory = True if use_gpu else False
        testset = ImageFolder(dataroot, transform=transform)
        # testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        print('LSUNCrop Test Num: ', len(testset))


class LSUN_Resize(object):
    def __init__(self, dataroot=CROOD_PATH + 'LSUN_resize', use_gpu=True, batch_size=128, img_size=32, options=None):
        transform =  test_transform(img_size)

        pin_memory = True if use_gpu else False
        testset = ImageFolder(dataroot, transform=transform)
        # testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        print('LSUNResize Test Num: ', len(testset))
