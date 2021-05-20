import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()

        # First convolution
        generator = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_channels, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            generator += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # 9 residual blocks
        for _ in range(9):
            generator += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            generator += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        generator += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_channels, 7),
                    nn.Tanh() ]

        self.generator = nn.Sequential(*generator)

    def forward(self, x):
        return self.generator(x)
