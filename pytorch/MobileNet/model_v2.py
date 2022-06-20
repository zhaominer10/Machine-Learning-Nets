from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tensorflow repo. 将卷积核的个数调整到8的整数倍
    It ensures that all layers have a channel number that is divisible by 8.
    (Maybe in order to adapt the number of hardware, e.g. the number of GPU)
    """
    if min_ch is None:
        min_ch = divisor

    # 得到离ch最近的8的整数倍的值
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # make sure that round down does not go down by more than 10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 创建基础卷积模块
# 有多种写法，参见：https://blog.csdn.net/qq_41488595/article/details/123305865
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) / 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 此处参考了pytorch与tensorflow官方对于mobilenet的实现，即对原论文中bottleneck为t=1(expand_ratio)的情况，invert residual中无第一个1x1卷积操作。
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # depth wise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
            # 因为最后一层卷积是使用线性激活函数y=x，即通过该线性激活函数，输入没有变化，所以此处就不用特别调用激活函数了
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):  # alpha是超参数，控制卷积核数量的倍率
        super(MobileNetV2, self).__init__()
        block = InvertResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)  # 第一层卷积核个数
        last_channel = _make_divisible(1208 * alpha, round_nearest)

        inverted_residual_bottlenecks = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []

        features.append(ConvBNReLU(3, input_channel, stride=2))

        for t, c, n, s in inverted_residual_bottlenecks:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
