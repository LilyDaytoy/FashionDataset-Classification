import torchvision.models as models
import torch.nn as nn


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
