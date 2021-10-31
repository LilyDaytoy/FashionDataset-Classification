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
import torch.optim as optim


class FashionDataset(Dataset):

    def __init__(self,
                 root: str = "../FashionDataset",
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


def load_data(root: str = "../FashionDataset",
              is_train: bool = True,
              is_val: bool = True,
              batch_size: int = 256,
              num_workers: int = 0
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


class ResNet50RemoveFC(nn.Module):
    def __init__(self):
        super(ResNet50RemoveFC, self).__init__()
        pretrained_net = models.resnet50(pretrained=True)
        modules = list(pretrained_net.children())[:-1]
        self.resnet_remove_fc = nn.Sequential(*modules)   # resnet_remove_fc是一个去掉fc层的resnet的所有modules的sequential module

    def forward(self, X):
        return self.resnet_remove_fc(X)


num_classes = [7, 3, 3, 4, 6, 3]
in_features = 2048


# 采用延后初始化，单独apply，在network里先不用对初始化作处理
class ResnetBranch6(nn.Module):
    def __init__(self):
        super(ResnetBranch6, self).__init__()
        self.pretrained_net_without_fc = ResNet50RemoveFC()
        self.fc = nn.ModuleList([nn.Linear(in_features, num_classes[0]),
                                 nn.Linear(in_features, num_classes[1]),
                                 nn.Linear(in_features, num_classes[2]),
                                 nn.Linear(in_features, num_classes[3]),
                                 nn.Linear(in_features, num_classes[4]),
                                 nn.Linear(in_features, num_classes[5])])

    def forward(self, X):
        output = []
        out_pooling = self.pretrained_net_without_fc(X)
        out_pooling_resized = out_pooling.view(-1, in_features)
        for i in range(6):
            output.append(self.fc[i](out_pooling_resized))
        return output


def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


device = torch.device('cpu')
dtype = torch.float32

attr_map = {"pattern": 0,
            "sleeve": 1,
            "length": 2,
            "neckline": 3,
            "material": 4,
            "tightness": 5}


def check_acc(model, data_loader):
    model.eval()
    num_correct = {"pattern": 0,
                   "sleeve": 0,
                   "length": 0,
                   "neckline": 0,
                   "material": 0,
                   "tightness": 0}
    num_samples = {"pattern": 0,
                   "sleeve": 0,
                   "length": 0,
                   "neckline": 0,
                   "material": 0,
                   "tightness": 0}

    with torch.no_grad():
        for (X, y) in data_loader:
            X = X.to(device=device, dtype=dtype)
            for attr in attr_map.keys():
                y[attr] = y[attr].to(device=device, dtype=torch.long)
            scores_list = model(X)
            preds = {}
            for attr, index in attr_map.items():
                _, preds[attr] = scores_list[index].max(dim=1)
            for attr in attr_map.keys():
                num_correct[attr] += (preds[attr] == y[attr]).sum()
                num_samples[attr] += torch.ones(preds[attr].size()).sum()
        for attr in attr_map.keys():
            print(attr,
                  f": num_correct/num_samples = {num_correct[attr]}/{num_samples[attr]}, "
                  f"accuracy = {float(num_correct[attr]) / num_samples[attr]}")


def train(model, train_loader, val_loader, optimizer, num_epochs, print_every):
    model = model.to(device=device)
    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device=device, dtype=dtype)
            for attr in attr_map.keys():   # dict不能直接to device
                y[attr] = y[attr].to(device=device, dtype=torch.long)

            model.train()
            # import pdb; pdb.set_trace();
            scores_list = model(X)
            scores_dict = {}
            for attr in attr_map.keys():
                scores_dict[attr] = scores_list[attr_map[attr]]
            loss = 0.0
            for attr, score in scores_dict.items():
                loss += F.cross_entropy(score, y[attr])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_every == 0:
                print(f"epoch {epoch}, iteration {i}, loss = {loss}")
                print("train accuracy: ")
                check_acc(model, train_loader)
                print("val accuracy: ")
                check_acc(model, val_loader)
                print()


# 训练
train_loader = load_data(root="../FashionDataset", batch_size=20)
val_loader = load_data(root="../FashionDataset", is_train=False, is_val=True, batch_size=20)
net = ResnetBranch6()
net.apply(xavier)  # 对所有fc层作延后初始化
learning_rate = 5e-5
params_1x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
params_fc = [param for name, param in net.named_parameters() if name in ["fc.weight", "fc.bias"]]
optimizer = optim.SGD([{'params': params_1x, 'lr': learning_rate},
                       {'params': params_fc, 'lr': learning_rate*10}])
train(model=net,
      train_loader=train_loader,
      val_loader=val_loader,
      optimizer=optimizer,
      num_epochs=10,
      print_every=20)


