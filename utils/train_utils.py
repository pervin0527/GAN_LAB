import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import utils as vutils

def save_fake_images(epoch, G, fixed_noise, args):
    with torch.no_grad():  # 그래디언트 계산을 하지 않음
        fake_images = G(fixed_noise).detach().cpu()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Fake Images at Epoch {epoch}")
    plt.imshow(np.transpose(vutils.make_grid(fake_images[:args.num_images], padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(f"{args.save_dir}/imgs/EP{epoch:>02}_Fake.png")  # 이미지 파일로 저장
    plt.close(fig)