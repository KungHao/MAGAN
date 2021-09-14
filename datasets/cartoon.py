import os
import math
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

class CartoonDataset(Dataset):
    def __init__(self, root, mode, selected_attrs, transform=None):
        self.items = self.make_dataset(root, mode, selected_attrs)
        self.root = root
        self.mode = mode
        self.transform = transform

    def make_dataset(self, root, mode, selected_attrs):
        assert mode in ['train', 'val', 'test']
        lines = [line.rstrip() for line in open(os.path.join(root, 'list_attr_cartoon.txt'), 'r')]
        all_attr_names = lines[1].split()
        attr2idx = {}
        idx2attr = {}
        for i, attr_name in enumerate(all_attr_names):
            attr2idx[attr_name] = i
            idx2attr[i] = attr_name

        lines = lines[2:]
        if mode == 'train':
            lines = lines[:-1000]       # train set leaves 1000 images
        if mode == 'val':
            lines = lines[-1000:-500]   # val set contains 20 images
        if mode == 'test':
            lines = lines[-500:]        # test set contains 1800 images

        items = []
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            label = []
            for attr_name in selected_attrs:
                idx = attr2idx[attr_name]
                label.append(values[idx] == '1')
            items.append([filename, label])
        return items

    def __getitem__(self, index):
        filename, label = self.items[index]
        image = Image.open(os.path.join(self.root, 'images', filename)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.items)

class CartoonDataLoader(object):
    def __init__(self, root, mode, selected_attrs, crop_size=None, image_size=128, batch_size=16):
        if mode not in ['train', 'test']:
            return

        transform = []
        if crop_size is not None:
            transform.append(transforms.CenterCrop(crop_size))
        transform.append(transforms.Resize(image_size))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        if mode == 'train':
            val_transform = transforms.Compose(transform)
            val_set = CartoonDataset(root, 'val', selected_attrs, transform=val_transform)
            self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            transform.insert(0, transforms.RandomHorizontalFlip())
            train_transform = transforms.Compose(transform)
            train_set = CartoonDataset(root, 'train', selected_attrs, transform=train_transform)
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))

        else:
            test_transform = transforms.Compose(transform)
            test_set = CartoonDataset(root, 'test', selected_attrs, transform=test_transform)
            self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))