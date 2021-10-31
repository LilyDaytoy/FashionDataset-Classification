import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tfm
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Callable, Optional
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FashionDataset(Dataset):

    def __init__(self,
                 root: str = "./FashionDataset",
                 is_train: bool = True,
                 is_val: bool = False,
                 transform: Optional[Callable] = None,
                 transform_target: Optional[Callable] = None
                 ) -> None:
        self.root = root
        self.is_train = is_train
        self.is_val = is_val
        self.transform = transform
        self.transform_target = transform_target
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
        self.img_file_path = os.path.join(root, img_file_name)
        self.data = []
        self.labels = {"pattern": [],
                       "sleeve": [],
                       "length": [],
                       "neckline": [],
                       "material": [],
                       "tightness": []}
        with open(self.img_file_path, 'r') as file:
            for line in file:
                self.data.append(line.rstrip("\n"))
        if label_file_name is not None:
            self.label_file_path = os.path.join(self.root, label_file_name)
            with open(self.label_file_path, 'r') as file:
                for line in file:
                    attributes = line.split(" ")
                    self.labels["pattern"].append(int(attributes[0]))
                    self.labels["sleeve"].append(int(attributes[1]))
                    self.labels["length"].append(int(attributes[2]))
                    self.labels["neckline"].append(int(attributes[3]))
                    self.labels["material"].append(int(attributes[4]))
                    self.labels["tightness"].append(int(attributes[5]))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.data[index])
        img = Image.open(img_path)
        if self.is_train:   # for training set
            if self.transform is not None:
                img = self.transform(img)
        else:    # for val set and test set
            if self.transform_target is not None:
                img = self.transform_target(img)
        if self.is_train or self.is_val:
            label = {}
            label["pattern"] = self.labels["pattern"][index]
            label["sleeve"] = self.labels["sleeve"][index]
            label["length"] = self.labels["length"][index]
            label["neckline"] = self.labels["neckline"][index]
            label["material"] = self.labels["material"][index]
            label["tightness"] = self.labels["tightness"][index]
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.data)


def load_data(root: str = "./FashionDataset",
              is_train: bool = True,
              is_val: bool = True,
              batch_size: int = 256,
              num_workers: int = 4
              ) -> DataLoader:
    normalize = tfm.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_aug = tfm.Compose([tfm.transforms.RandomResizedCrop(224),
                             tfm.transforms.RandomHorizontalFlip(),
                             tfm.transforms.ToTensor(),
                             normalize])
    test_aug = tfm.Compose([tfm.Resize(256),
                            tfm.CenterCrop(224),
                            tfm.ToTensor(),
                            normalize])
    dataset = FashionDataset(root=root, is_train=is_train, is_val=is_val, transform=train_aug, transform_target=test_aug)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


device = torch.device("cuda")
dtype = torch.floar32
num_classes = {"pattern": 7,
               "sleeve": 3,
               "length": 3,
               "neckline": 4,
               "material": 6,
               "tightness": 3}


def check_acc(model, data_loader, attribute):
    num_correct = 0
    num_samples = 0
    for X, y in data_loader:
        X = X.to(device=device,dtype=dtype)
        y = y[attribute].to(device=device, dtype=torch.long)
        model.eval()
        scores = model(X)
        _, preds = scores.max(dim=1)
        num_correct += (preds == y).sum()
        num_samples += torch.ones(preds.size()).sum()
    print(f'num_correct/num_samples = {num_correct}/{num_samples}, accuracy = {float(num_correct) / num_samples}')


def train(attribute, model, train_loader, val_loader, optimizer, num_epochs = 10, print_every = 10):
    model = model.to(device=device)
    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(train_loader):
            # y is a dictionary with 6 attributes, each attribute has a batch_size of data
            X = X.to(device=device, dtype=dtype)
            y = y[attribute].to(device=device, dtype=torch.long)

            model.train()
            scores = model(X)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_every == 0:
                print(f"epoch {epoch}, iteration {i}, loss = {loss}")
                print("train accuracy: ")
                check_acc(model, train_loader, attribute=attribute)
                print("val accuracy: ")
                check_acc(model, val_loader, attribute=attribute)
                print()


# pattern
learning_rate = 5e-5
# re-initialize the last fc layer with the corresponding num_classes of the specific attribute
pretrained_net = models.resnet50(pretrained=True)
pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, num_classes["pattern"])
nn.init.xavier_uniform_(pretrained_net.fc.weight)
params_1x = [param for name, param in pretrained_net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
optimizer = torch.optim.SGD([{'params': params_1x},
                 {'params': pretrained_net.fc.parameters(), 'lr': learning_rate * 10}], lr=learning_rate, weight_decay=0.001)
train_loader = load_data(batch_size=20)
val_loader = load_data(is_train=False, is_val=True, batch_size=20)
train(pretrained_net, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, num_epochs=10)


