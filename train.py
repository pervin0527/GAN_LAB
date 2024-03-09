import os
import torch

from torch import nn
from tqdm import tqdm
from torchsummary import summary
from torch.utils.data import DataLoader

from utils.basic_utils import Args
from data.dataset import get_dataset
from models.model_loader import load_model
from utils.train_utils import save_fake_images


def train(D, G, dataloader, d_optimizer, g_optimizer, criterion, args):
    D_losses, G_losses = [], []
    D_real_accuracies, D_fake_accuracies_before_update, D_fake_accuracies_after_update = [], [], []
    
    for idx, (images, _) in enumerate(tqdm(dataloader, desc="Train", leave=False)):
        bs = images.size(0)
        real_images = images.to(args.device)
        real_labels = torch.full((bs,), 1., dtype=torch.float, device=args.device)
        fake_labels = torch.full((bs,), 0., dtype=torch.float, device=args.device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        ## first term of Object function
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        Dx = D(real_images).view(-1) ## [128, 1, 1, 1] --> [128]
        d_real_loss = criterion(Dx, real_labels)
        D_real_accuracies.append((Dx > 0.5).float().mean().item())

        ## second term of Object function
        z = torch.randn(bs, args.nz, 1, 1, device=args.device) ## [batch_size, 100, 1, 1]
        Gz = G(z) ## fake images [batch_size, 3, 64, 64]

        ## detach : 생성자 G의 출력에서 계산된 그래디언트가 생성자 자체로 역전파되는 것을 방지함.
        DGz1 = D(Gz.detach()).view(-1) ## [128, 1, 1, 1] --> [128]
        d_fake_loss = criterion(DGz1, fake_labels) 

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        D_fake_accuracies_before_update.append((DGz1 < 0.5).float().mean().item()) ## 확률이 낮을수록 가짜에 속지 않음.

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        ## Train Generator
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        z = torch.randn(bs, args.nz, 1, 1, device=args.device) ## [batch_size, 100, 1, 1]
        DGz2 = D(Gz).view(-1) ## [128, 1, 1, 1] --> [128]
        # g_loss = -criterion(DGz2, real_labels)
        g_loss = criterion(DGz2, real_labels) ## 생성한 데이터를 진짜라고 구분할수록 오차가 더 낮음.
        g_loss.backward()
        g_optimizer.step()
        D_fake_accuracies_after_update.append((DGz2 > 0.5).float().mean().item()) ## 확률이 높을수록 가짜를 진짜로 판별한 것.

        D_losses.append(d_loss.item())
        G_losses.append(g_loss.item())

    avg_metrics = {
        'D_loss': sum(D_losses) / len(D_losses),
        'G_loss': sum(G_losses) / len(G_losses),
        'D_real_acc': sum(D_real_accuracies) / len(D_real_accuracies),
        'D_fake_acc_before': sum(D_fake_accuracies_before_update) / len(D_fake_accuracies_before_update),
        'D_fake_acc_after': sum(D_fake_accuracies_after_update) / len(D_fake_accuracies_after_update),
    }
    return avg_metrics


def main():
    args = Args("./hyps.yaml")
    args.num_workers = os.cpu_count()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    G, D = load_model(args)
    D = D.to(args.device)
    G = G.to(args.device)

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.d_lr)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.g_lr)

    fixed_noise = torch.randn(args.num_images, args.nz, 1, 1, device=args.device)
    print("\nStart Training.")
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        metrics = train(D, G, dataloader, d_optimizer, g_optimizer, criterion, args)
        print(f'Discriminator Loss: {metrics["D_loss"]:.4f}')
        print(f'Generator Loss: {metrics["G_loss"]:.4f}')
        print(f'Discriminator Real Accuracy: {metrics["D_real_acc"]:.4f}')
        print(f'Discriminator Fake Accuracy (Before G Update): {metrics["D_fake_acc_before"]:.4f}')
        print(f'Discriminator Fake Accuracy (After G Update): {metrics["D_fake_acc_after"]:.4f}\n') ##  판별자가 가짜 이미지를 "진짜"로 잘못 분류한 점수.

        save_fake_images(epoch+1, G, fixed_noise, args)

    # Save the model checkpoints 
    torch.save(G.state_dict(), f'{args.save_dir}/ckpt/G.ckpt')
    torch.save(D.state_dict(), f'{args.save_dir}/ckpt/D.ckpt')


if __name__ == "__main__":
    main()