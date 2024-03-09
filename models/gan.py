from torch import nn

class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nc = args.nc
        self.img_size = args.img_size
        self.D = nn.Sequential(nn.Linear(self.nc * self.img_size * self.img_size, self.ndf),
                               nn.ReLU(),
                               nn.Linear(self.ndf, self.ndf),
                               nn.Dropout(0.3),
                               nn.ReLU(),
                               nn.Dropout(0.3),
                               nn.Linear(self.ndf, 1),
                               nn.Sigmoid())
        
    def forward(self, x):
        x = x.view(-1, self.nc * self.img_size * self.img_size)
        x = self.D(x)

        return x
    

class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nc = args.nc
        self.ngf = args.ngf
        self.img_size = args.img_size
        
        self.G = nn.Sequential(
            nn.Linear(self.nz, self.ngf),
            nn.ReLU(),
            nn.Linear(self.ngf, self.ngf),
            nn.ReLU(),
            nn.Linear(self.ngf, self.nc * self.img_size * self.img_size),
            # nn.Tanh()
            nn.Sigmoid()
        )
        
    def forward(self, z):
        z = z.view(z.size(0), -1) # z를 [batch_size, args.nz]로 평탄화
        z = self.G(z)
        z = z.view(-1, self.nc, self.img_size, self.img_size) # 출력을 [batch_size, nc, img_size, img_size]로 변환
        return z