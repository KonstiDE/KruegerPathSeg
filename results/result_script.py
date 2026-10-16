import shutup

from loss.dice_sb_loss import DiceSBLoss

shutup.please()

import os.path

import matplotlib.pyplot as plt
import numpy as np

import torch
import config.config as cfg

from tqdm import tqdm

from provider.dataset_provider import get_loader

from model.dense121_unet.dense121_unet_model import DenseNet121
from model.unet.unet_model import UNET

import statistics as s

from metrics.mean_errors import mae, r_mse, medae
from metrics.similarity import ssim, zncc
from metrics.topological import completeness, correctness, quality
from metrics.literature import calculate_metrics
from metrics.counter import positive_pixels

device = "cuda:0"


def test(model, checkpoint_path, images, version, variant):
    torch.cuda.empty_cache()

    loss_fn = DiceSBLoss()

    result_path = os.path.join(checkpoint_path, "results")

    if os.path.isdir(checkpoint_path) and not os.path.isdir(result_path):
        os.mkdir(result_path)

    torch.no_grad()

    loader = get_loader(cfg.get_test_path(version), 1, shuffle=False)

    loop = tqdm(loader)

    running_loss = []
    running_mae = []
    running_medae = []
    running_rmse = []
    running_zncc = []
    running_ssim = []

    running_comp = []
    running_corr = []
    running_qual = []

    running_prec = []
    running_reca = []
    running_f1sc = []
    running_accu = []
    running_ioum = []

    running_positive = []

    for (data, target, _, _, file) in loop:
        data = data.to(device)
        target = target.to(device)

        prediction, _, _ = model(data)
        prediction = torch.sigmoid(prediction)

        # loss = loss_fn(prediction, target)
        # _, rmse = r_mse(prediction, target)
        #
        # running_loss.append(loss.item())
        # running_mae.append(mae(prediction, target).item())
        # running_medae.append(medae(prediction, target).item())
        # running_rmse.append(rmse.item())
        # running_zncc.append(zncc(prediction, target).item())
        # running_ssim.append(ssim(prediction, target).item())
        #
        # running_comp.append(completeness(prediction, target).item())
        # running_corr.append(correctness(prediction, target).item())
        # running_qual.append(quality(prediction, target).item())
        #
        # literature_metrics = calculate_metrics(prediction, target)
        # running_prec.append(literature_metrics["precision"])
        # running_reca.append(literature_metrics["recall"])
        # running_f1sc.append(literature_metrics["f1"])
        # running_accu.append(literature_metrics["accuracy"])
        # running_ioum.append(literature_metrics["iou"])

        running_positive.append(positive_pixels(prediction).item())

        if images:
            prediction = prediction.squeeze(0).squeeze(0).detach().cpu()
            target = target.squeeze(0).squeeze(0).detach().cpu()

            data = data.squeeze(0).cpu().numpy()
            red = data[0]
            red_normalized = (red * (255 // red.max())).astype(np.uint8)
            green = data[1]
            green_normalized = (green * (255 // green.max())).astype(np.uint8)
            blue = data[2]
            blue_normalized = (blue * (255 // blue.max())).astype(np.uint8)

            beauty = np.dstack((red_normalized, green_normalized, blue_normalized))

            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

            im = axs[0].imshow(beauty)
            axs[0].set_xticklabels([])
            axs[0].set_yticklabels([])
            # plt.colorbar(im, ax=axs[0])

            im = axs[1].imshow(prediction, cmap="viridis")
            axs[1].set_xticklabels([])
            axs[1].set_yticklabels([])
            # im.set_clim(0, max(prediction.max(), target.max()))
            plt.colorbar(im, ax=axs[1])

            im = axs[2].imshow(target, cmap="viridis")
            axs[2].set_xticklabels([])
            axs[2].set_yticklabels([])
            plt.colorbar(im, ax=axs[2])

            plt.savefig(os.path.join(result_path, file[0].replace("df_", "").replace(".npz", "") + ".png"))
            plt.close(fig)

    print("Variant: {}, version: {}".format(variant, version))
    # print("LOSS\t{}".format(round(s.mean(running_loss), 4)))
    # print("MAE\t{}".format(round(s.mean(running_mae), 4)))
    # print("MedAE\t{}".format(round(s.mean(running_medae), 4)))
    # print("RMSE\t{}".format(round(s.mean(running_rmse), 4)))
    # print("SSIM\t{}".format(round(s.mean(running_ssim), 4)))
    # print("ZNCC\t{}".format(round(s.mean(running_zncc), 4)))
    # print("")
    # print("COMP\t{}".format(round(s.mean(running_comp), 4)))
    # print("CORR\t{}".format(round(s.mean(running_corr), 4)))
    # print("QUAL\t{}".format(round(s.mean(running_qual), 4)))
    #print("---Now literature metrics---")
    #print("Precision {}".format(round(s.mean(running_prec), 4)))
    #print("Recall {}".format(round(s.mean(running_reca), 4)))
    #print("F1 {}".format(round(s.mean(running_f1sc), 4)))
    #print("Accuracy {}".format(round(s.mean(running_accu), 4)))
    #print("IoU {}".format(round(s.mean(running_ioum), 4)))

    print("Positive pixels: {}".format(sum(running_positive)))
    print("Positive pixels mean: {}".format(s.mean(running_positive)))
    print("Positive pixels median: {}".format(s.median(running_positive)))

    # with open("{}_{}_lit.txt".format(variant, version), "w+") as file:
    #     file.write("{}\n".format(s.mean(running_prec)))
    #     file.write("{}\n".format(s.mean(running_reca)))
    #     file.write("{}\n".format(s.mean(running_f1sc)))
    #     file.write("{}\n".format(s.mean(running_accu)))
    #     file.write("{}\n".format(s.mean(running_ioum)))
    #
    #     file.close()


if __name__ == '__main__':
    immense_config = [
        ["unet", False, False, False, "buffer", "checkpoint_000015", "model_epoch16.pt"],
        ["unet", False, False, False, "cluster", "checkpoint_000016", "model_epoch17.pt"],
        ["denseunet", False, False, False, "buffer", "checkpoint_000010", "model_epoch11.pt"],
        ["denseunet", False, False, False, "cluster", "checkpoint_000009", "model_epoch10.pt"],
        ["aspp", True, False, False, "buffer", "checkpoint_000007", "model_epoch8.pt"],
        ["aspp", True, False, False, "cluster", "checkpoint_000009", "model_epoch10.pt"],
        ["attention", False, True, False, "buffer", "checkpoint_000010", "model_epoch11.pt"],
        ["attention", False, True, False, "cluster", "checkpoint_000015", "model_epoch16.pt"],
        ["stacked", False, False, True, "buffer", "checkpoint_000009", "model_epoch10.pt"],
        ["stacked", False, False, True, "cluster", "checkpoint_000009", "model_epoch10.pt"]
    ]

    for single_config in immense_config:
        variant, aspp, attention, aux, version, checkpoint, epoch_str = single_config

        base_path = "E:/master/ray_results/runs_5e-05/seed/final/" + variant + "/" + version + "/" + checkpoint + "/"

        if variant == "unet":
            model = UNET(in_channels=3, out_channels=1).cuda()
        else:
            model = DenseNet121(in_channels=3, out_channels=1, aspp=aspp, attention=attention, aux=aux).cuda()

        state_dict = torch.load(os.path.join(
            base_path + epoch_str
        ))["net_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()

        test(
            model,
            base_path,
            images=False,
            version=version,
            variant=variant
        )
