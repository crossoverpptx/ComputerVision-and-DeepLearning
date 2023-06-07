import torch.nn as nn
import torch


# 作用：将输入特征矩阵深度调整为divisor即这里的8的整数倍
# 更好地调用硬件设备
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    # int(ch + divisor / 2) // divisor：相当于一个四舍五入的操作
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 普通卷积结构
class ConvReLU(nn.Sequential):
    # 输入特征矩阵深度，输出特征矩阵深度，卷积核大小，步距，groups=1是普通卷积；groups=in_channel是dw卷积
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, groups=1):
        super(ConvReLU, self).__init__(
            # bias偏置不使用，因为下面要使用BN层
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.ReLU6(inplace=True)
        )


# 注意力模块
class CBAM(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 通道注意力机制
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel // reduction, out_features=in_channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # 空间注意力机制
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # 通道注意力机制
        maxout = self.max_pool(x)
        maxout = self.mlp(maxout.view(maxout.size(0), -1))
        avgout = self.avg_pool(x)
        avgout = self.mlp(avgout.view(avgout.size(0), -1))
        channel_out = self.sigmoid(maxout + avgout)
        channel_out = channel_out.view(x.size(0), x.size(1), 1, 1)
        channel_out = channel_out * x
        # 空间注意力机制
        max_out, _ = torch.max(channel_out, dim=1, keepdim=True)
        mean_out = torch.mean(channel_out, dim=1, keepdim=True)
        out = torch.cat((max_out, mean_out), dim=1)
        out = self.sigmoid(self.conv(out))
        out = out * channel_out
        return out


# 二阶项残差
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out1 = self.relu(out)
        out2 = self.relu(identity)
        out = out + identity + torch.sqrt(out1*out2+0.0001)
        return out


# MSCNA模型
class MSCNA(nn.Module):
    # round_nearest=8
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MSCNA, self).__init__()
        input_channel = _make_divisible(32 * alpha, round_nearest)
        print(input_channel)
        # last_channel = _make_divisible(1280 * alpha, round_nearest)

        # conv1
        self.conv11 = ConvReLU(3, input_channel, 3, padding=1)
        self.conv12 = ConvReLU(3, input_channel, 5, padding=2)
        self.conv13 = ConvReLU(3, input_channel, 7, padding=3)

        # conv2
        conv2 = ConvReLU(32, 64, 3)
        features1 = [conv2]
        self.features1 = nn.Sequential(*features1)

        # pool1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3
        features2 = [ConvReLU(64, 128, 3)]
        self.features2 = nn.Sequential(*features2)

        # self.avg_pool = nn.AvgPool2d(2)
        # self.max_pool = nn.MaxPool2d(2)

        # 注意力模块
        features31 = [CBAM(128)]
        self.features31 = nn.Sequential(*features31)

        # 二阶项残差模块
        features32 = [BasicBlock(128, 128)]
        self.features32 = nn.Sequential(*features32)

        # conv4
        features3 = [ConvReLU(128, 256, 3)]
        self.features3 = nn.Sequential(*features3)

        # pool2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv5-7
        features4 = [ConvReLU(256, 512, 3), ConvReLU(512, 512, 3),
                     ConvReLU(512, 512, 3)]
        self.features4 = nn.Sequential(*features4)

        # pool3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv8-10
        features5 = [ConvReLU(512, 512, 3), ConvReLU(512, 512, 3),
                     ConvReLU(512, 512, 3)]
        self.features5 = nn.Sequential(*features5)

        # pool4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        # weight initialization
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    # nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv11(x) + self.conv12(x) + self.conv13(x)
        x = self.features1(x)
        x = self.pool1(x)
        x = self.features2(x)

        x = self.features31(x) + self.features32(x)

        x = self.features3(x)
        x = self.pool2(x)
        x = self.features4(x)
        x = self.pool3(x)
        x = self.features5(x)
        x = self.pool4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
