import torch


def positive_pixels(prediction, threshold=0.5):
    prediction[prediction < 0.5] = 0

    return torch.sum(prediction)
