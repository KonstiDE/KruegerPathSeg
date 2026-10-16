import torch


def completeness(prediction, target, threshold=0.5):
    prediction = (prediction >= threshold).float() if threshold is not None else prediction
    TP = torch.sum(prediction * target)
    FN = torch.sum((1 - prediction) * target)
    return TP / (TP + FN + 1e-6)


def correctness(prediction, target, threshold=0.5):
    prediction = (prediction >= threshold).float() if threshold is not None else prediction
    TP = torch.sum(prediction * target)
    FP = torch.sum(prediction * (1 - target))
    return TP / (TP + FP + 1e-6)


def quality(prediction, target, threshold=0.5):
    corr = correctness(prediction, target, threshold)
    comp = completeness(prediction, target, threshold)

    return (corr * comp) / ((comp + corr) * (comp - corr) + 1e-06)
