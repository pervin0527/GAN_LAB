import torch
import importlib

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def load_model(args):
    model_name = args.model_name.lower()
    model_module = importlib.import_module(f"models.{model_name}")

    if model_name in ["wgan"]:
        Generator = getattr(model_module, "Generator")
        Discriminator = getattr(model_module, "Critic")

    else:
        Generator = getattr(model_module, "Generator")
        Discriminator = getattr(model_module, "Discriminator")

    G = Generator(args)
    D = Discriminator(args)

    if args.init_weights:
        G.apply(weights_init)
        D.apply(weights_init)

    return G, D