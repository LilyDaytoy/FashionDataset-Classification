import torchvision.models as models
import torch.nn as nn


class ResNet50RemoveFC(nn.Module):
    def __init__(self):
        super(ResNet50RemoveFC, self).__init__()
        pretrained_net = models.resnet50(pretrained=True)
        modules = list(pretrained_net.children())[:-1]
        self.resnet_remove_fc = nn.Sequential(*models)

    def forward(self, X):
        return self.resnet_remove_fc(X)


num_classes = {"pattern": 7,
               "sleeve": 3,
               "length": 3,
               "neckline": 4,
               "material": 6,
               "tightness": 3}
in_features = 2048


# 采用延后初始化，单独apply，在network里先不用对初始化作处理
class ResnetBranch6(nn.Module):
    def __init__(self):
        self.pretrained_net_without_fc = ResNet50RemoveFC()
        self.fc = {}
        self.fc["pattern"] = nn.Linear(in_features, num_classes["pattern"])
        self.fc["sleeve"] = nn.Linear(in_features, num_classes["sleeve"])
        self.fc["length"] = nn.Linear(in_features, num_classes["length"])
        self.fc["neckline"] = nn.Linear(in_features, num_classes["neckline"])
        self.fc["material"] = nn.Linear(in_features, num_classes["material"])
        self.fc["tightness"] = nn.Linear(in_features, num_classes["tightness"])

    def forward(self, X):
        output = []
        for attr in num_classes.keys():
            net_attr = nn.Sequential(self.pretrained_net_without_fc, self.fc[attr])
            output.append(net_attr(X))
        return output


def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
