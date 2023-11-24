import pyro
import pyro.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class DigitPositions(BaseModel):
    def __init__(self, z_where_dim=2):
        super().__init__()
        self.mu = torch.zeros(z_where_dim)
        self.sigma = torch.ones(z_where_dim) * 0.2

    def forward(self, t, z_where=None):
        sigma = self.sigma
        if z_where is None:
            sigma = sigma * 5
        prior = dist.Normal(self.mu, self.sigma).to_event(1)
        return pyro.sample("z_where_%d" % t, prior)

class DigitFeatures(BaseModel):
    def __init__(self, z_what_dim=10):
        super().__init__()
        self._dim = z_what_dim

    def forward(self, K=3):
        prior = dist.Normal(0, 1).expand([K, self._dim]).to_event(2)
        return pyro.sample("z_what", prior)

class DigitsDecoder(BaseModel):
    def __init__(self, digit_side=28, hidden_dim=400, x_side=96, z_what_dim=10):
        super().__init__()
        self._digit_side = digit_side
        self._x_side = x_side
        self.decoder = nn.Sequential(
            nn.Linear(z_what_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, digit_side ** 2), nn.Sigmoid()
        )
        self.scale = torch.diagflat(torch.ones(2) * x_side / digit_side)
        self.translate = (x_side - digit_side) / digit_side

    def blit(self, digits, z_where):
        S, B, K, _ = z_where.shape
        affine_p1 = self.scale.repeat(S, B, K, 1, 1)
        affine_p2 = z_where.unsqueeze(-1) * self.translate
        affine_p2[:, :, :, 0, :] = -affine_p2[:, :, :, :, 0, :]
        grid = affine_grid(
            torch.cat((affine_p1, affine_p2), -1).view(S*B*K, 2, 3),
            torch.Size((S*B*K, 1, self._x_side, self._x_side)),
            align_corners=True
        )

        digits = digits.view(S*B*K, self._digit_side, self._digit_side)
        frames = grid_sample(digits.unsqueeze(1), grid, mode='nearest',
                             align_corners=True).squeeze(1)
        return frames.view(S, B, K, self._x_side, self._x_side)

    def forward(self, t, what, where, x):
        digits = self.decoder(z_what)
        frame = torch.clamp(self.blit(digits, z_where).sum(-2), 0., 1.)
        likelihood = dist.ContinuousBernoulli(frame).to_event(1)
        return pyro.sample("X_%d" % t, likelihood, obs=x)

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
