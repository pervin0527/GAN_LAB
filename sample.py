import os
import torch

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.basic_utils import Args
from models.gan import get_D_and_G
from data.dataset import get_dataset
from utils.train_utils import save_fake_images

def train(D, G, dataloader, d_optimizer, g_optimizer, criterion, args):
    fixed_noise = torch.randn(args.num_images, args.nz, 1, 1, device=args.device)
    for epoch in range(args.epochs):
        for idx, (images, _) in enumerate(tqdm(dataloader, desc="Train", leave=False)):
            bs = images.size(0)
            images = images.reshape(bs, -1).to(args.device)
            real_labels = torch.ones(bs, 1).to(args.device)
            fake_labels = torch.zeros(bs, 1).to(args.device)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            outputs = D(images)
            d_real_loss = criterion(outputs, real_labels) ## first term of object func
            real_score = outputs

            z = torch.randn(bs, args.nz).to(args.device)
            fake_images = G(z)
            outputs = D(fake_images)
            d_fake_loss = criterion(outputs, fake_labels) ## second_term of object func
            fake_score = outputs

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            z = torch.randn(bs, args.nz).to(args.device)
            fake_images = G(z)
            outputs = D(fake_images)
            g_loss = criterion(outputs, real_labels) ## 가짜 데이터를 입력 받았을 때 D가 진짜라고 분류한 비율을 이용해 오차를 계산한다.
            g_loss.backward()
            g_optimizer.step()

        print(f"\nEpoch[{epoch}/{args.epochs}], Step[{idx+1}/{len(dataloader)}]")
        print(f"D_loss : {d_loss.item():.4f}, G_loss : {g_loss.item():.4f}")
        print(f"Real Score : {real_score.mean().item():.4f}, Fake Score : {fake_score.mean().item():.4f}")
        
        # Save sampled images
        save_fake_images(epoch+1, G, fixed_noise, args)

    # Save the model checkpoints 
    torch.save(G.state_dict(), f'{args.save_dir}/G.ckpt')
    torch.save(D.state_dict(), f'{args.save_dir}/D.ckpt')
                                    

def main():
    args = Args("./hyps.yaml")
    args.num_workers = os.cpu_count()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    D, G = get_D_and_G(args)
    D = D.to(args.device)
    G = G.to(args.device)

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.d_lr)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.g_lr)
    train(D, G, dataloader, d_optimizer, g_optimizer, criterion, args)

if __name__ == "__main__":
    main()