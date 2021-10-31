from torch.utils.data import DataLoader
from torchvision import transforms as tfm
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Callable, Optional
import matplotlib.pyplot as plt


class FashionDataset(Dataset):

    def __init__(self,
                 is_train: bool = True,
                 is_val: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ) -> None:
        self.root = "../FashionDataset"
        self.is_train = is_train
        self.is_val = is_val
        self.transform = transform
        self.target_transform = target_transform
        if self.is_train:
            img_file_name = "split/train.txt"
            label_file_name = "split/train_attr.txt"
        else:
            if self.is_val:
                img_file_name = "split/val.txt"
                label_file_name = "split/val_attr.txt"
            else:
                img_file_name = "split/test.txt"
                label_file_name = None
        self.img_file_path = os.path.join(self.root, img_file_name)
        self.data = []
        self.labels = []
        with open(self.img_file_path, 'r') as file:
            for line in file:
                self.data.append(line.rstrip('\n'))
        if label_file_name is not None:
            self.label_file_path = os.path.join(self.root, label_file_name)
            with open(self.label_file_path, 'r') as file:
                for line in file:
                    for x in [int(x) for x in line.split(" ")]:
                        self.labels.append(x)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.data[index])
        image = Image.open(image_path)
        if self.is_train or self.is_val:
            if self.transform is not None:
                image = self.transform(image)
        else:
            if self.target_transform is not None:
                image = self.target_transform(image)
        if self.is_train or self.is_val:
            return {"image": image, "label": self.labels[index]}
        else:
            return image

    def __len__(self):
        return len(self.data)


def load_data(is_train: bool = True,
              is_val: bool = False,
              batch_size: int = 256) -> DataLoader:
    train_aug = tfm.Compose([tfm.Resize(256), tfm.RandomCrop(224), tfm.RandomHorizontalFlip(), tfm.ToTensor()])
    test_aug = tfm.Compose([tfm.Resize(256), tfm.RandomCrop(224), tfm.ToTensor()])
    dataset = FashionDataset(is_train=is_train, is_val=is_val, transform=train_aug, target_transform=test_aug)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def show_image(index, dataset):  # the tool to display a single image
    image = dataset[index]['image'].permute(1, 2, 0)
    plt.imshow(image)


