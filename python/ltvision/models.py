import litetorch as lt
from litetorch import nn


class Sequential(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []

    def add(self, module):
        self.layers.append(module)
        return self

    def forward(self, x):
        for m in self.layers:
            x = m.forward(x)
        return x

    def parameters(self):
        ps = []
        for m in self.layers:
            ps = ps + m.parameters()
        return ps

    def to(self, device):
        for m in self.layers:
            m.to(device)



def _conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride, 1, False)


def _conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, 0, False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.conv2 = _conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c * self.expansion:
            self.downsample = Sequential()
            self.downsample.add(_conv1x1(in_c, out_c * self.expansion, stride))
            self.downsample.add(nn.BatchNorm2d(out_c * self.expansion))

    def forward(self, x):
        identity = x
        out = self.relu.forward(self.bn1.forward(self.conv1.forward(x)))
        out = self.bn2.forward(self.conv2.forward(out))
        if self.downsample is not None:
            identity = self.downsample.forward(x)
        return self.relu.forward(lt.Ops.add(out, identity))

    def parameters(self):
        ps = self.conv1.parameters() + self.bn1.parameters() + self.conv2.parameters() + self.bn2.parameters()
        if self.downsample is not None:
            ps = ps + self.downsample.parameters()
        return ps

    def to(self, device):
        for m in (self.conv1, self.bn1, self.conv2, self.bn2):
            m.to(device)
        if self.downsample is not None:
            self.downsample.to(device)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = _conv1x1(in_c, out_c)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = _conv3x3(out_c, out_c, stride)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = _conv1x1(out_c, out_c * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_c * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or in_c != out_c * self.expansion:
            self.downsample = Sequential()
            self.downsample.add(_conv1x1(in_c, out_c * self.expansion, stride))
            self.downsample.add(nn.BatchNorm2d(out_c * self.expansion))

    def forward(self, x):
        identity = x
        out = self.relu.forward(self.bn1.forward(self.conv1.forward(x)))
        out = self.relu.forward(self.bn2.forward(self.conv2.forward(out)))
        out = self.bn3.forward(self.conv3.forward(out))
        if self.downsample is not None:
            identity = self.downsample.forward(x)
        return self.relu.forward(lt.Ops.add(out, identity))

    def parameters(self):
        ps = (self.conv1.parameters() + self.bn1.parameters() + self.conv2.parameters()
              + self.bn2.parameters() + self.conv3.parameters() + self.bn3.parameters())
        if self.downsample is not None:
            ps = ps + self.downsample.parameters()
        return ps

    def to(self, device):
        for m in (self.conv1, self.bn1, self.bn2, self.conv3, self.bn3):
            m.to(device)
        if self.downsample is not None:
            self.downsample.to(device)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_c, n, stride=1):
        seq = Sequential()
        seq.add(block(self.in_c, out_c, stride))
        self.in_c = out_c * block.expansion
        for _ in range(1, n):
            seq.add(block(self.in_c, out_c))
        return seq

    def forward(self, x):
        x = self.relu.forward(self.bn1.forward(self.conv1.forward(x)))
        x = self.maxpool.forward(x)
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        x = self.layer4.forward(x)
        x = self.flatten.forward(self.avgpool.forward(x))
        return self.fc.forward(x)

    def parameters(self):
        ps = []
        for m in (self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4, self.fc):
            ps = ps + m.parameters()
        return ps

    def to(self, device):
        for m in (self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4, self.fc):
            m.to(device)


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(7, 7)
        self.flatten = nn.Flatten()
        self.classifier = Sequential()
        self.classifier.add(nn.Linear(512 * 7 * 7, 4096))
        self.classifier.add(nn.ReLU())
        self.classifier.add(nn.Dropout())
        self.classifier.add(nn.Linear(4096, 4096))
        self.classifier.add(nn.ReLU())
        self.classifier.add(nn.Dropout())
        self.classifier.add(nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features.forward(x)
        x = self.flatten.forward(self.avgpool.forward(x))
        return self.classifier.forward(x)

    def parameters(self):
        return self.features.parameters() + self.classifier.parameters()

    def to(self, device):
        self.features.to(device)
        self.classifier.to(device)


def _make_vgg_layers(cfg):
    seq = Sequential()
    in_c = 3
    for v in cfg:
        if v == "M":
            seq.add(nn.MaxPool2d(2, 2, 0))
        else:
            seq.add(nn.Conv2d(in_c, v, 3, 1, 1))
            seq.add(nn.ReLU())
            in_c = v
    return seq


_VGG_CFGS = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg11(num_classes=1000):
    return VGG(_make_vgg_layers(_VGG_CFGS["vgg11"]), num_classes)


def vgg13(num_classes=1000):
    return VGG(_make_vgg_layers(_VGG_CFGS["vgg13"]), num_classes)


def vgg16(num_classes=1000):
    return VGG(_make_vgg_layers(_VGG_CFGS["vgg16"]), num_classes)


def vgg19(num_classes=1000):
    return VGG(_make_vgg_layers(_VGG_CFGS["vgg19"]), num_classes)
