from torch import nn

class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nz = args.nz
        self.ngf = args.ngf
        self.nc = args.nc

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nz, out_channels=self.ngf * 8, kernel_size=4, stride=1, padding=0, bias=False), ## latent --> (self.ngf*8) x 4 x 4
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, stride=2, padding=1, bias=False), ## (self.ngf*8) x 4 x 4 --> (self.ngf*4) x 8 x 8
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False), ## (self.ngf*4) x 8 x 8 --> (self.ngf*2) x 16 x 16
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1, bias=False), ## (self.ngf*2) x 16 x 16 --> (self.ngf) x 32 x 32
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=self.nc, kernel_size=4, stride=2, padding=1, bias=False), ## (self.ngf) x 32 x 32 --> (self.nc) x 64 x 64
            nn.Tanh())

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nc = args.nc
        self.ndf = args.ndf

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=False), ## 3 x 64 x 64 --> ndf x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), ## ndf x 32 x 32 --> (ndf * 2) x 16 x 16
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), ## (ndf * 2) x 16 x 16 --> (ndf * 4) x 8 x 8
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf * 4, out_channels=self.ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), ## (ndf * 4) x 8 x 8 --> (ndf*8) x 4 x 4
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)