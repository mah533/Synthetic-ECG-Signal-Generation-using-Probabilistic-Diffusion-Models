"""
from GitHub: https://github.com/mah533/Augmentation-of-ECG-Training-Dataset-with-CGAN/blob/main/model_wgan_gp_ecg.py
"""
import torch
import torch.nn as nn

class Disc_dcgan_gp_1d(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64          16x1x256
            nn.Conv1d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),  # 16x64x128
            nn.LeakyReLU(0.2),
            ## _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),  # 16x128x64
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x256x32
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 16x512x16
            nn.Conv1d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),  # 16x1x7
            # nn.Linear(7, 1)         # 16x1x1,   3 for E5000 and 7 for MIT-BIH 256
            nn.Linear(1, 1)           # 16x1x1,   1 for MIT-BIH 64
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False,),
            # since BN is used, no need to bias
            nn.InstanceNorm1d(out_channels, affine=True),  # LayerNorm ←→ InstanceNorm
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Gen_dcgan_gp_1d(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        self.gen = nn.Sequential(
            # input: N x z_dim x 1      BATCH_SIZEx100x1
            self._block(z_dim, features_g * 16, 4, 1, 0),   # BSxfeatures_gx4x4           BSx1024x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8               BSx512x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16              BSx256x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32              BSx128x32
            nn.ConvTranspose1d(
               features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
             ),                                         # BSx1x64
            # nn.Linear(64, 256, bias=False),           # BSx1x256
            nn.Linear(64, 64, bias=False),              # BSx1x256
            nn.Tanh(),                                  # BSx1x64 [-1, 1]

        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

"""
def runn():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Disc_dcgan_gp_1d(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Gen_dcgan_gp_1d(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn(N, z_dim, 1, 1)
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success!")


runn()
"""