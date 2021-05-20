import torch.nn as nn
import torch.nn.functional as func

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        # 3 convolution layers
        # Instance normalizations
        # LeakyReLU
        discriminator = [nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        discriminator += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        discriminator += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        discriminator += [nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # The classification layer
        discriminator += [nn.Conv2d(512, 1, 4, padding=1)]
        self.discriminator = nn.Sequential(*discriminator)

    def forward(self, x):
        x =  self.discriminator(x)
        # Average pooling and flatten
        return func.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
