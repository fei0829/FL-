import torch
from torchvision.models.vgg import VGG, make_layers, cfgs


class MLP_MNIST(torch.nn.Module):
    # MLP
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class CNN_MNIST(torch.nn.Module):
    # MLP
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, (3, 3))
        self.conv2 = torch.nn.Conv2d(16, 32, (4, 4))
        self.fc1 = torch.nn.Linear(32 * 25, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.conv1(x)), (2, 2))  # 16 * 13 * 13
        x = torch.max_pool2d(torch.relu(self.conv2(x)), (2, 2))  # 32 * 5 * 5
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class CNN_CIFAR10(torch.nn.Module):
    # MLP
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (3, 3))
        self.conv2 = torch.nn.Conv2d(16, 32, (4, 4))
        self.conv3 = torch.nn.Conv2d(32, 64, (3, 3))
        self.fc1 = torch.nn.Linear(64 * 2 * 2, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), (2, 2)))  # 64 * 15 * 15
        x = torch.relu(torch.max_pool2d(self.conv2(x), (2, 2)))  # 128 * 6 * 6
        x = torch.relu(torch.max_pool2d(self.conv3(x), (2, 2)))  # 256 * 2 * 2
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class MLP_CIFAR10(torch.nn.Module):
    # MLP
    def __init__(self):
        super(MLP_CIFAR10, self).__init__()
        self.fc1 = torch.nn.Linear(3 * 32 * 32, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


def vgg11():
    model = VGG(make_layers(cfgs["A"], batch_norm=True), num_classes=10)
    return model


