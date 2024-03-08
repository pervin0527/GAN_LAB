from torch import nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, img_size):
        super().__init__()
        self.nc = nc
        self.img_size = img_size
        self.D = nn.Sequential(nn.Linear(nc * img_size * img_size, ndf),
                               nn.LeakyReLU(0.2),
                               nn.Linear(ndf, ndf),
                               nn.LeakyReLU(0.2),
                               nn.Linear(ndf, 1),
                               nn.Sigmoid())
        
    def forward(self, x):
        x = x.view(-1, self.nc * self.img_size * self.img_size)
        x = self.D(x)

        return x
    

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, img_size):
        super().__init__()
        self.nc = nc
        self.img_size = img_size
        
        self.G = nn.Sequential(
            nn.Linear(nz, ngf),
            nn.ReLU(),
            nn.Linear(ngf, ngf),
            nn.ReLU(),
            nn.Linear(ngf, nc * img_size * img_size),
            nn.Tanh()
        )
        
    def forward(self, z):
        z = z.view(z.size(0), -1) # z를 [batch_size, args.nz]로 평탄화
        z = self.G(z)
        z = z.view(-1, self.nc, self.img_size, self.img_size) # 출력을 [batch_size, nc, img_size, img_size]로 변환
        return z


def get_D_and_G(args):
    D = Discriminator(args.nc, args.ndf, args.img_size)
    G = Generator(args.nz, args.ngf, args.nc, args.img_size)

    D.apply(weights_init)
    G.apply(weights_init)

    return D, G