import torch
from .registry import Registry

META_ARCHITECTURE = Registry(name="ARCHITECTURE")


def build_model(config, vocab):
    model = META_ARCHITECTURE.get(config.model.architecture)(config.model, vocab)
    model = model.to(torch.device(config.model.device))

    return model
