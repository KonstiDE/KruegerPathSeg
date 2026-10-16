import torch


def calculate_metrics(prediction, target, threshold=0.5):
    prediction = (prediction >= threshold).float() if threshold is not None else prediction

    prediction = prediction.view(-1)
    target = target.view(-1)

    TP = torch.sum((prediction == 1) & (target == 1)).item()
    FP = torch.sum((prediction == 1) & (target == 0)).item()
    FN = torch.sum((prediction == 0) & (target == 1)).item()
    TN = torch.sum((prediction == 0) & (target == 0)).item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "accuracy": accuracy,
        "iou": iou
    }
