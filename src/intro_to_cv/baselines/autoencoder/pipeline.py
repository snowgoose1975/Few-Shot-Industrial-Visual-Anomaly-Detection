"""CPU model loading and reconstruction inference shared by AE commands."""

import torch
from intro_to_cv.baselines.autoencoder.model import Autoencoder
from intro_to_cv.common.images import image_tensor
from intro_to_cv.common.scoring import top_fraction_mean


def load_model(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Autoencoder().cpu()
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state["size"]


def infer(model, size, path):
    x = image_tensor(path, size).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(x)
    error = (x - reconstructed).abs().mean(dim=1)[0].numpy()
    values = error.ravel()
    score = top_fraction_mean(values)
    return score, error, reconstructed[0].permute(1, 2, 0).numpy()

