import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import paddle.fluid.dygraph as dg


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return dg.Conv2D(in_planes, out_planes, filter_size=3, stride=stride,
                     padding=dilation, groups=groups, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return dg.Conv2D(in_planes, out_planes, filter_size=3, stride=stride, padding=1)


class Bottleneck(fluid.dygraph.Layer):
    '''
    DNN网络
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = dg.BatchNorm
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = fluid.layers.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = fluid.layers.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = fluid.layers.relu(out)

        return out


class MyCNN(fluid.dygraph.Layer):
    def __init__(self, block, layers, num_classes=65,
                groups=1, width_per_group=64):
        super(MyCNN, self).__init__()
        norm_layer = dg.BatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = dg.Conv2D(1, self.inplanes, filter_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(self.inplanes, param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
        self.maxpool = dg.Pool2D(pool_size=3, pool_stride=2, pool_padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = dg.Pool2D(5, pool_type="avg")
        self.fc = Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = dg.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, self.dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return dg.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = fluid.layers.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = fluid.layers.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def make_model(num_classes=65):
    layers = [2, 2]
    return MyCNN(Bottleneck, layers, num_classes=num_classes)