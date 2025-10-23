import torch
# import torch_dataset_mirror
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
import sys


class CustomDataset(Dataset):
    def __init__(self, df, target_column):
        self.labels = torch.tensor(df[target_column].values, dtype=torch.int64)
        self.features = torch.tensor(df.drop(target_column, axis=1).values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data_mlp(dataset,batch_size,dim_factor=2):
    if dataset == 'EMNIST':
        # EMNIST train dataset
        train_loader = torch.utils.data.DataLoader(datasets.EMNIST(
            root='./data',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1751,), (0.3332,))
            ]),
            download=True,
            split='balanced'),
            batch_size=batch_size,
            shuffle=True)

        # EMNIST test dataset
        test_loader = torch.utils.data.DataLoader(datasets.EMNIST(
            root='./data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1751,), (0.3332,))
            ]),
            download=True,
            split='balanced'),
            batch_size=batch_size,
            shuffle=False)

        indim = 784
        outdim = 47
        

    elif dataset == "MNIST":
        # MNIST train dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=batch_size, shuffle=True)
        # MNIST test dataset
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=False)
        indim = 784
        outdim = 10


    elif dataset == "Fashion_MNIST":
        # Fashion MNIST train dataset
        train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
            root='./data',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            download=True),
            batch_size=batch_size,
            shuffle=True)
        
        # Fashion MNIST test dataset
        test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
            root='./data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            download=True),
            batch_size=batch_size,
            shuffle=False)
        indim = 784
        outdim = 10
    
    elif dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])),
            batch_size=batch_size, shuffle=False)
        indim = 3072
        outdim = 10
        
        
    elif dataset == "CIFAR100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762])])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762])])),
            batch_size=batch_size, shuffle=False)
        indim = 3072
        outdim = 100



    elif dataset == "FER2013":
        train_loader = torch.utils.data.DataLoader(
            datasets.FER2013(
            root='./data',
            split='train',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5077,), (0.2550,))
            ])),
        batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.FER2013(
            root='./data',
            split='test',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5077,), (0.2550,))
            ]),
        ),
        batch_size=batch_size, shuffle=False)
        indim = 48*48
        outdim = 7     

    elif dataset == "SVHN":
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
            root='./data/svhn',
            split='train',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4377, 0.4438, 0.4728],
                                    [0.198, 0.201, 0.197])
            ]),
            download=True
        ),
        batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
            root='./data/svhn',
            split='test',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4377, 0.4438, 0.4728],
                                    [0.198, 0.201, 0.197])
            ]),
            download=True
        ),
        batch_size=batch_size, shuffle=False)
        indim = 32*32*3
        outdim = 10              

    elif dataset == "HIGGS":
        csv_file = "./data/HIGGS.csv"
        column_names = ["outcome"] + ["feature "+str(i) for i in range(1,29)]
        df = pd.read_csv(csv_file, header=None, names=column_names)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        X_df = df.iloc[:, 1:]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        df = pd.DataFrame(np.concatenate([df.iloc[:, :1].values, X_scaled], axis=1), columns=['outcome'] + list(df.columns[1:]))
        train_df, test_df = np.split(df.sample(frac=1, random_state=42), [int(0.8*len(df))])
        train_dataset = CustomDataset(train_df, "outcome")
        test_dataset = CustomDataset(test_df, "outcome")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        indim = 28
        outdim = 2

    elif dataset == 'tin':
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2296, 0.2263, 0.2255]
        train_loader = torch.utils.data.DataLoader(TinyImageNet_load(root="./data/tiny-imagenet-200/", train=True, transform=transforms.Compose([
                # transforms.RandomResizedCrop(64),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(TinyImageNet_load('./data/tiny-imagenet-200/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])),
            batch_size=batch_size, shuffle=False)
        indim = 64*64*3
        outdim = 200           

    elif dataset == "SUSY":
        csv_file = "./data/SUSY.csv"
        # SUSY dataset has 18 features + 1 label column
        # First column is the label (signal=1, background=0)
        # Remaining 18 columns are features
        column_names = ["label"] + ["feature_" + str(i) for i in range(1, 19)]
        df = pd.read_csv(csv_file, header=None, names=column_names)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        # Separate features and normalize them
        X_df = df.iloc[:, 1:]  # Features (columns 1-18)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        
        # Reconstruct dataframe with normalized features
        df = pd.DataFrame(
            np.concatenate([df.iloc[:, :1].values, X_scaled], axis=1), 
            columns=['label'] + list(df.columns[1:])
        )
        
        # Split into train/test (80/20 split)
        train_df, test_df = np.split(df.sample(frac=1, random_state=42), [int(0.8*len(df))])
        
        train_dataset = CustomDataset(train_df, "label")
        test_dataset = CustomDataset(test_df, "label")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        indim = 18
        outdim = 2

    dimension = indim * dim_factor
    hiddim = [dimension, dimension, dimension]
    
    return train_loader, test_loader, indim, outdim, hiddim


class TinyImageNet_load(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt