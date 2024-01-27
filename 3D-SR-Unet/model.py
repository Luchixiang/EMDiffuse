import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom


class CubicWeightedPSNRLoss(nn.Module):
    def __init__(self):
        super(CubicWeightedPSNRLoss, self).__init__()

    def forward(self, upsampled_input, pred, target):
        # Perform cubic upsampling on the input
        error = (upsampled_input - target) ** 2
        weight = error / (error.max() * 2) + 0.5
        # Compute the pixel-wise cubic-weighted MSE loss
        weighted_mse = ((pred - target) ** 2 * weight).mean()
        # Compute the cubic-weighted PSNR loss
        return weighted_mse


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def conv3x3x3(in_channels, out_channels, stride=1,
              padding=1, bias=True, groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


class SRUNet(nn.Module):
    def __init__(self, up_scale=6):
        super().__init__()
        self.up_scale = up_scale
        self.conv1_1 = conv3x3x3(1, 32)
        self.conv1_2 = conv3x3x3(32, 32)
        self.conv1_3 = conv3x3x3(32, 32)
        self.fracconv1 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3,
                                            stride=(self.up_scale, 1, 1), padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2_1 = conv3x3x3(32, 64)
        self.conv2_2 = conv3x3x3(64, 64)
        self.conv2_3 = conv3x3x3(64, 64)
        self.fracconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, stride=(2, 1, 1), padding=1)
        self.conv3_1 = conv3x3x3(64, 128)
        self.conv3_2 = conv3x3x3(128, 128)
        self.conv3_3 = conv3x3x3(128, 128)
        self.fracconv3 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=(2, 2, 2),
                                            padding=1)
        self.conv2_4 = conv3x3x3(128, 64)
        self.conv2_5 = conv3x3x3(64, 64)
        self.conv2_6 = conv3x3x3(64, 64)
        self.fracconv4 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3,
                                            stride=(self.up_scale // 2, 2, 2), padding=1)
        self.conv1_4 = conv3x3x3(64, 32)
        self.conv1_5 = conv3x3x3(32, 32)
        self.conv1_6 = conv3x3x3(32, 32)
        self.final_conv = conv3x3x3(32, 1)

    def forward(self, x):
        x_1_1 = F.relu(self.conv1_1(x))
        x_1_2 = F.relu(self.conv1_2(x_1_1))
        x_1_3 = F.relu(self.conv1_3(x_1_2))
        b, c, d, h, w = x_1_3.shape
        x_frac1 = self.fracconv1(x_1_3, output_size=(b, c, d * self.up_scale, h, w))
        # print(x_frac1.shape)
        x_2_1 = self.pool(x_1_3)
        x_2_2 = F.relu(self.conv2_1(x_2_1))

        x_2_3 = F.relu(self.conv2_2(x_2_2))
        x_2_4 = F.relu(self.conv2_3(x_2_3))
        b, c, d, h, w = x_2_4.shape
        x_frac2 = self.fracconv2(x_2_4, output_size=(b, c, d * 2, h, w))
        # print(x_frac2.shape)
        x_3_1 = self.pool(x_2_4)
        x_3_2 = F.relu(self.conv3_1(x_3_1))
        x_3_3 = F.relu(self.conv3_2(x_3_2))
        x_3_4 = F.relu(self.conv3_3(x_3_3))
        b, c, d, h, w = x_3_4.shape
        x_frac3 = self.fracconv3(x_3_4, output_size=(b, c, d * 2, h * 2, w * 2))
        # print(x_frac3.shape)
        x_merge_2 = torch.concatenate([x_frac3, x_frac2], dim=1)
        x_2_5 = F.relu(self.conv2_4(x_merge_2))
        x_2_6 = F.relu(self.conv2_5(x_2_5))
        x_2_7 = F.relu(self.conv2_6(x_2_6))
        b, c, d, h, w = x_2_7.shape
        x_frac4 = self.fracconv4(x_2_7, output_size=(b, c, d * self.up_scale // 2, h * 2, w * 2))
        # print(x_frac4.shape)
        x_merge_1 = torch.concatenate([x_frac1, x_frac4], dim=1)
        x_1_4 = F.relu(self.conv1_4(x_merge_1))
        x_1_5 = F.relu(self.conv1_5(x_1_4))
        x_1_6 = F.relu(self.conv1_6(x_1_5))
        out = self.final_conv(x_1_6)
        return out


if __name__ == '__main__':
    model = SRUNet(up_scale=6)

    test_gt = torch.rand((1, 1, 16, 128, 128))
    test_input = torch.rand((1, 1, 16, 128, 128))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    test_out = model(test_input)
    loss_function = CubicWeightedPSNRLoss()
    loss = loss_function(test_input, test_gt)
    optimizer.zero_grad()
    loss.backward()
