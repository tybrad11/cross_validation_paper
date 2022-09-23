
from torch import nn
import torch

class NNModel(nn.Module):
    def __init__(self, num_classes: int = 10,
                 image_size = (28,28),
                 hidden_sizes = (128,64),
                 output_size=10) -> None:
        super(NNModel, self).__init__()

        self.nn = nn.Sequential(nn.Linear(image_size[0]*image_size[1], hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0],-1)
        x = self.nn(x)
        return x


class ConvModel(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(ConvModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(8, 32, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def select_model(selection):
  if selection == 'ConvModel':
    return ConvModel()
  if selection == 'NNModel':
    return NNModel()

