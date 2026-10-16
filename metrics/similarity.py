import torch

from pytorch_msssim import ssim as ssim_new


def ssim(pred, target):
    data_range = max(pred.max() - pred.min(), target.max() - target.min()).item()

    return ssim_new(pred, target, data_range=data_range, win_size=5)


def zncc(pred, target, eps=0.00001):
    znccs = 0
    numel = 1 / torch.numel(pred / len(pred))
    for i in range(len(pred)):
        p = pred[i]
        t = target[i]

        avg1 = torch.mean(p)
        avg2 = torch.mean(t)

        first = 1 / (torch.std(p) * torch.std(t) + eps)

        znccs += torch.mul(torch.sum(torch.mul(first, torch.mul(torch.sub(p, avg1), torch.sub(t, avg2)))), numel)
    return znccs


if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)

    a = torch.sigmoid(torch.randn(8, 1, 512, 512)).cuda()
    b = torch.sigmoid(torch.randn(8, 1, 512, 512)).cuda()

    data_range = max(a.max() - a.min(), b.max() - b.min()).item()

    print(ssim(a, b))
