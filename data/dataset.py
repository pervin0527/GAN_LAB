import torch
import torchvision

from torchvision import transforms

def get_transform(args):
    transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(args.mean, args.std)])
    
    return transform


def get_dataset(args):
    if args.mean is None and args.std is None and args.dataset_name != "mnist":
        args.mean = (0.485, 0.456, 0.406)
        args.std = (0.229, 0.224, 0.225)
    
    elif args.dataset_name == "mnist":
        args.mean = [0.5]
        args.std = [0.5]

    transform = get_transform(args)
    
    if args.dataset_name.lower() == "mnist":
        dataset = torchvision.datasets.MNIST(args.data_dir, train=True, transform=transform, download=True)

    elif args.dataset_name.lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10(args.data_dir, train=True, transform=transform, download=True)

    else:
        dataset = torchvision.datasets.ImageFolder(root=f"{args.data_dir}/{args.dataset_name}", transform=transform)

    return dataset
