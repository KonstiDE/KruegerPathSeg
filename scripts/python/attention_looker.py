import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import config as cfg
from model.dense121_unet.dense121_unet_model import DenseNet121
from provider.dataset_provider import get_loader

if __name__ == '__main__':
    s = torch.load("E:/master/ray_results/runs_5e-05/seed/final/stacked/buffer/checkpoint_000009/model_epoch10.pt")["net_state_dict"]

    m = DenseNet121(aux=True).cuda()

    m.load_state_dict(s)

    loader = get_loader(cfg.get_test_path("buffer"), 1, shuffle=True, load_amount=100)

    loop = tqdm(loader)

    i = 0
    for (data, target, degree, distance, file) in loop:
        if i == 25:
            data = data.cuda()
            target = target.cuda()

            prediction, deg, dis = m(data)
            prediction = torch.sigmoid(prediction)

            plt.imshow(deg.squeeze(0).squeeze(0).detach().cpu().numpy())
            plt.show()

            plt.imshow(dis.squeeze(0).squeeze(0).detach().cpu().numpy())
            plt.show()

            print(deg.shape)
            print(dis.shape)

            exit(2)

        i += 1
