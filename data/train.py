from data.net import ResnetBranch6
from data.load_data import load_data
from data.net import xavier
import torch.optim as optim
import torch
import torch.nn.functional as F


device = torch.device('cuda')
dtype = torch.float32

attr_map = {"pattern": 0,
            "sleeve": 1,
            "length": 2,
            "neckline": 3,
            "material": 4,
            "tightness": 5}


def check_acc(model, data_loader):
    model.eval()
    num_correct = {}
    num_samples = {}

    with torch.no_grad():
        for (X, y) in data_loader:
            X = X.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
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
            y = y.to(device=device, dtype=torch.long)

            model.train()
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
train_loader = load_data(batch_size=20)
val_loader = load_data(is_train=True, is_val=False, batch_size=20)
net = ResnetBranch6()
net.apply(xavier)  # 对所有fc层作延后初始化
learning_rate = 5e-5
params_1x = [param for param, name in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
params_fc = [param for param, name in net.named_parameters() if name in ["fc.weight", "fc.bias"]]
optimizer = optim.SGD([{'params': params_1x, 'lr': learning_rate},
                       {'params': params_fc, 'lr': learning_rate*10}])
train(model=net,
      train_loader=train_loader,
      val_loader=val_loader,
      optimizer=optimizer,
      num_epochs=10,
      print_every=20)


