import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from torch.nn import init


if isinstance(torch.fft, types.ModuleType):
    class FFT_Mask_ForBack(torch.nn.Module):
        def __init__(self):
            super(FFT_Mask_ForBack, self).__init__()
        def forward(self, x, full_mask):
            full_mask = full_mask[..., 0]
            x_in_k_space = torch.fft.fft2(x)
            masked_x_in_k_space = x_in_k_space * full_mask.view(1, 1, *(full_mask.shape))
            masked_x = torch.real(torch.fft.ifft2(masked_x_in_k_space))
            return masked_x
else:
    class FFT_Mask_ForBack(torch.nn.Module):
        def __init__(self):
            super(FFT_Mask_ForBack, self).__init__()

        def forward(self, x, mask):
            x_dim_0 = x.shape[0]
            x_dim_1 = x.shape[1]
            x_dim_2 = x.shape[2]
            x_dim_3 = x.shape[3]
            x = x.view(-1, x_dim_2, x_dim_3, 1)
            y = torch.zeros_like(x)
            z = torch.cat([x, y], 3)
            fftz = torch.fft(z, 2)
            z_hat = torch.ifft(fftz * mask, 2)
            x = z_hat[:, :, :, 0:1]
            x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
            return x


# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))


        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):

        x = PhiTb

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, self.fft_forback, PhiTb, mask)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]