import torch.nn as nn
from torchvision.models.video.resnet import Bottleneck, BasicBlock, BasicStem, Conv3DSimple, Conv3DNoTemporal, \
    r2plus1d_18, mc3_18
import torchvision.models.video as video_models
from torchvision.models.video import r3d_18

class CustomBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(CustomBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.InstanceNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.InstanceNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class OutputBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(OutputBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            # nn.InstanceNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            # nn.InstanceNorm3d(planes)
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)

        return out


class CustomStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(CustomStem, self).__init__(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.InstanceNorm3d(64), # no batch norm if batch size is 1
            nn.ReLU(inplace=True))


class FCNClassifVideoResNet(nn.Module):

    def __init__(self,
                 block=CustomBlock,
                 conv_makers=[Conv3DSimple] * 4,
                 layers=[2, 2, 2, 2],
                 stem=CustomStem,
                 num_classes=5,
                 zero_init_residual=False
                 ):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(FCNClassifVideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.layerconv1d_1 = self._make_layer(OutputBlock, conv_makers[0], num_classes, layers[3], stride=1)
        self.layerconv1d_2 = self._make_layer(OutputBlock, conv_makers[0], num_classes, layers[3], stride=1)
        self.maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        # x = x.flatten(1)
        # x = self.fc(x)
        x = self.layerconv1d_1(x)
        # x = self.layerconv1d_2(x)
        x = self.maxpool(x)
        x = x.flatten(1)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.InstanceNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.InstanceNorm3d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class FC3DResNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        # self.backbone = r3d_18(pretrained=pretrained)
        # self.backbone = r2plus1d_18(pretrained=pretrained)
        self.backbone = mc3_18(pretrained=pretrained)

        # Replace input conv to accept 1 channel instead of 3
        old_conv1 = self.backbone.stem[0]
        self.backbone.stem[0] = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )

        # Replace fc with conv + pooling
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Conv3d(512, num_classes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone.stem(x)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)  # shape: [B, 512, T', H', W']

        out = self.classifier(features)  # shape: [B, num_classes, T', H', W']
        out = self.pool(out)  # shape: [B, num_classes, 1, 1, 1]
        return out.view(out.size(0), -1)  # shape: [B, num_classes]
