"""
various blocks
"""
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.autograd as autograd
from torch.autograd import Variable


class Classif_FC5_BN(nn.Module):
    # myClassifier01
    # FC 5 layers
    def __init__(self, signal_len, num_classes):
        super().__init__()

        self.signal_len = signal_len
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(self.signal_len, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(4096, self.num_classes),
            nn.Tanh()

        )

    def forward(self, z, batch_size, device):
        scores = self.model(z)
        return scores


class Classif_LSTM(nn.Module):
    # myClassifier02
    def __init__(self, inp_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.n_layers = num_layers
        self.hidden_dim = hidden_dim
        # self.out_dim  = out_dim
        self.lstm = nn.LSTM(input_size=inp_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, input, batch_size, device):
        h_0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim, dtype=torch.float).to(device)
        c_0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim, dtype=torch.float).to(device)
        input = input.reshape(batch_size, 1, 256)
        output, (h_o, c_o) = self.lstm(input, (h_0, c_0))
        out = self.tanh(self.fc(output.squeeze()))

        return out  # out.reshape((out.size(0), 1, out.size(-1)))


def normalize(x):
    # Ming, Li, 2014, "Verification Based ECG Biometrics with ..."
    x_min = min(x)
    x_max = max(x)
    x_norm = [2 * (x[i] - (x_max + x_min) / 2) / (x_max - x_min) for i in range(len(x))]
    return x_norm


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # fake = Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
