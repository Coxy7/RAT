import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __getitem__(self, index):
        item = self.dataset[index]
        return item, index

    def __len__(self):
        return len(self.dataset)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, with_index=False, **kwargs):
        if with_index:
            dataset = DatasetWithIndex(dataset)
        super().__init__(dataset, **kwargs)


def get_dataloader(dataset, shuffle=False, drop_last=False, with_index=False, num_replicas=1, rank=0, **kwargs):
    if with_index:
        dataset = DatasetWithIndex(dataset)
    if num_replicas > 1:
        sampler = DistributedSampler(
            dataset, num_replicas, rank, shuffle=shuffle, drop_last=drop_last)
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, **kwargs)
    else:
        loader = torch.utils.data.DataLoader(
            dataset, shuffle=shuffle, drop_last=drop_last, **kwargs)
    return loader


def stratified_random_split(dataset, labels, train_split, val_split=None, seed=7):
    indices = list(range(len(dataset)))
    indices_train, indices_test = train_test_split(
        indices,
        train_size=train_split,
        random_state=seed,
        stratify=labels,
    )
    if val_split is not None:
        labels_test = [labels[i] for i in indices_test]
        indices_val, indices_test = train_test_split(
            indices_test,
            train_size=(val_split / (1 - train_split)),
            random_state=seed,
            stratify=labels_test,
        )
    trainset = Subset(dataset, indices_train)
    testset = Subset(dataset, indices_test)
    if val_split is not None:
        valset = Subset(dataset, indices_val)
        return trainset, valset, testset
    return trainset, testset


class BaseDataset():
    
    def __init__(self, size=None, mean=None, std=None):
        self.size = size
        self.mean = mean
        self.std = std

    def get_loader(self, data_dir, batch_size, num_workers, with_index=False):
        raise NotImplementedError()
    
    def preprocess(self, images):
        if self.mean:
            return TF.normalize(images, self.mean, self.std)
        return images


class ImageFolderDataset(BaseDataset):
    
    def __init__(self, folder, domain, num_classes):
        super().__init__(
            size=224,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        self.num_classes = num_classes
        self.folder = folder
        self.domain = domain
        self.transforms_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(self.mean, self.std)
        ])
        self.transforms_test = transforms.Compose([  # SRDC
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.Normalize(self.mean, self.std)
        ])
        # self.transforms_train = self.transforms_test

    def get_loader(self, data_dir, batch_size, num_workers, with_index=False,
                   train_split='none', val_split=None, split_seed=0,
                   augment=True, drop_last=True, num_devices=1, rank=0):
        if type(batch_size) is int:
            bs_train, bs_test = batch_size, batch_size
        else:
            bs_train, bs_test = batch_size
        data_dir = os.path.join(data_dir, self.folder, self.domain)

        transforms_train = self.transforms_train if augment else self.transforms_test
        transforms_test = self.transforms_test

        if train_split == 'none':
            aug_trainset = ImageFolder(root=data_dir, transform=transforms_train)
            raw_trainset = ImageFolder(root=data_dir, transform=transforms_test)
            testset = ImageFolder(root=data_dir, transform=transforms_test)
        else:
            train_split = float(train_split)
            aug_dataset = ImageFolder(root=data_dir, transform=transforms_train)
            raw_dataset = ImageFolder(root=data_dir, transform=transforms_test)
            labels = [label for _, label in aug_dataset.imgs]
            aug_trainset, _ = stratified_random_split(
                aug_dataset, labels, train_split, seed=split_seed)
            if val_split:
                raw_trainset, valset, testset = stratified_random_split(
                    raw_dataset, labels, train_split, val_split=val_split, seed=split_seed)
            else:
                raw_trainset, testset = stratified_random_split(
                    raw_dataset, labels, train_split, seed=split_seed)

        aug_trainloader = get_dataloader(
            aug_trainset, batch_size=bs_train, shuffle=True,
            drop_last=drop_last, num_workers=num_workers, with_index=with_index,
            num_replicas=num_devices, rank=rank)
        raw_trainloader = get_dataloader(
            raw_trainset, batch_size=bs_train, shuffle=False,
            drop_last=False, num_workers=num_workers, with_index=with_index,
            num_replicas=num_devices, rank=rank)
        testloader = get_dataloader(
            testset, batch_size=bs_test, shuffle=False,
            drop_last=False, num_workers=num_workers, with_index=with_index)
        if val_split:
            valloader = get_dataloader(
                valset, batch_size=bs_test, shuffle=False,
                drop_last=False, num_workers=num_workers, with_index=with_index)
            return aug_trainloader, raw_trainloader, valloader, testloader

        return aug_trainloader, raw_trainloader, testloader

    def preprocess(self, images):
        x = images
        x = TF.normalize(x, self.mean, self.std)
        return x
