import torch
import torch.nn as nn
import torch.nn.functional as functional

from torch.autograd import Variable
from torch.nn.init import dirac, kaiming_normal


def dirac_delta(ni, no, k):
    n = min(ni, no)
    size = (n, n) + k
    repeats = (max(no // ni, 1), max(ni // no, 1)) + (1,) * len(k)
    return dirac(torch.Tensor(*size)).repeat(*repeats)


def normalize(w):
    """Normalizes weight tensor over full filter."""
    return functional.normalize(w.view(w.size(0), -1)).view_as(w)


class DiracConv2d(nn.Conv2d):
    """Dirac parametrized convolutional layer.

    Works the same way as `nn.Conv2d`, but has additional weight parametrizatoin:
        :math:`\alpha\delta + \beta W`,
    where:
        :math:`\alpha` and :math:`\beta` are learnable scalars,
        :math:`\delta` is such a tensor so that `F.conv2d(x, delta) = x`, ie
            Kroneker delta
        `W` is weight tensor

    It is user's responsibility to set correcting padding. Only stride=1 supported.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=True):
        super(DiracConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride=1, padding=padding, dilation=dilation, bias=bias)
        self.alpha = nn.Parameter(torch.Tensor([5]))
        self.beta = nn.Parameter(torch.Tensor([1e-5]))
        self.register_buffer('delta', dirac_delta(in_channels, out_channels, self.weight.size()[2:]))
        assert self.delta.size() == self.weight.size()

    def forward(self, input):
        return functional.conv2d(input, self.alpha * Variable(self.delta) + self.beta * normalize(self.weight),
                        self.bias, self.stride, self.padding, self.dilation)


class NCRelu(nn.Module):
    def __init__(self, inplace=False):
        super(NCRelu, self).__init__()
        self.inplace=inplace

    def forward(self, x):
        return ncrelu(x)

    def __repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
        + inplace_str + ')'


def ncrelu(x):
    return torch.cat([x.clamp(min=0),
                      x.clamp(max=0)], dim=1)


# diracconv2d -> NCRelu -> BN  x = delta*x
def add_DiracConv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False, layer=3):
    dim_out = dim_out // 2
    if useBN:
        if layer == 3:
            return nn.Sequential(
                DiracConv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out*2),
                DiracConv2d(dim_out*2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out*2),
            )
        elif layer == 4:
            return nn.Sequential(
                DiracConv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out*2),
                DiracConv2d(dim_out*2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out*2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2)
            )
        elif layer == 5:
            return nn.Sequential(
                DiracConv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out*2),
                DiracConv2d(dim_out*2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out*2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2)
            )
        elif layer == 6:
            return nn.Sequential(
                DiracConv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out*2),
                DiracConv2d(dim_out*2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out*2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2)
            )
        elif layer == 10:
            return nn.Sequential(
                DiracConv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out*2),
                DiracConv2d(dim_out*2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out*2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2),
                DiracConv2d(dim_out * 2, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
                NCRelu(),
                nn.BatchNorm2d(dim_out * 2)

            )

    else:
        return nn.Sequential(
            DiracConv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ReLU(),
            DiracConv2d(dim_out, dim_out, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ReLU()
        )


def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
    if useBN:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )


def add_merge_stage(ch_coarse, ch_fine, in_coarse, in_fine, upsample):
    conv = nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
    torch.cat(conv, in_fine)

    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
    )
    upsample(in_coarse)


def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )


class Net(nn.Module):
    def __init__(self, useBN=False, layer=3):
        super(Net, self).__init__()

        self.conv1 = add_conv_stage(1, 32, useBN=useBN)
        self.conv2 = add_conv_stage(32, 64, useBN=useBN)
        self.conv3 = add_conv_stage(64, 128, useBN=useBN)
        self.conv4 = add_conv_stage(128, 256, useBN=useBN)
        self.conv5 = add_conv_stage(256, 512, useBN=useBN)

        self.conv4m = add_conv_stage(512, 256, useBN=useBN)
        self.conv3m = add_conv_stage(256, 128, useBN=useBN)
        self.conv2m = add_conv_stage(128,  64, useBN=useBN)
        self.conv1m = add_conv_stage( 64,  32, useBN=useBN)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128,  64)
        self.upsample21 = upsample(64 ,  32)

    # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        conv1_out = self.conv1(x)
        # return self.upsample21(conv1_out)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv5_out = self.conv5(self.max_pool(conv4_out))

        conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
        conv4m_out = self.conv4m(conv5m_out)

        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)

        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)

        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)

        conv0_out = self.conv0(conv1m_out)

        return conv0_out
